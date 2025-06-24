"""
PyTorch dataset classes for datasets in the NICE-SLAM format.
Large chunks of code stolen and adapted from:
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from openfungraph.utils.colmap import read_model
from openfungraph.utils.general_utils import to_scalar, measure_time


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def as_intrinsics_matrix_rotated(intrinsics, ori_height):
    """
    Get matrix representation of intrinsics.
    [self.fx, self.fy, self.cx, self.cy]

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[1]
    K[1, 1] = intrinsics[0]
    K[0, 2] = ori_height - intrinsics[3]
    K[1, 2] = intrinsics[2]
    return K


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            print(len(self.color_paths))
            print(len(self.depth_paths))
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.camera_axis == 'Left':
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if self.camera_axis == 'Left':
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth = np.expand_dims(depth, -1) 
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        if not hasattr(self, 'camera_axis'):
            self.camera_axis = 'Up'
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        if self.camera_axis == 'Left':
            K = as_intrinsics_matrix_rotated([self.fx, self.fy, self.cx, self.cy], self.desired_height)
        elif self.camera_axis == 'Up':
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        if self.camera_axis == 'Left':
            K = datautils.scale_intrinsics(
                K, self.width_downsample_ratio, self.height_downsample_ratio
            )
        elif self.camera_axis == 'Up':
            K = datautils.scale_intrinsics(
                K, self.height_downsample_ratio, self.width_downsample_ratio
            )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        # print(intrinsics)

        pose = self.transformed_poses[index]
        # print(pose)

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )


class SceneFun3DDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "lowres_wide.traj")
        # self.pose_path = os.path.join(self.input_folder, "hires_poses.traj")
        meta_file = os.path.join(basedir, 'metadata.csv')
        with open(meta_file, encoding = 'utf-8') as f:
            meta_csv = np.loadtxt(f,str,delimiter = ",")
        for line in meta_csv:
            if sequence.split('/')[0] in line and sequence.split('/')[1] in line:
                self.camera_axis = line[2]
        # self.camera_axis = 'Up'
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/wide/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/highres_depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        poses_from_traj = {}
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = np.array(self.TrajStringToMatrix(line)[1].tolist())
        color_paths, depth_paths, embedding_paths = self.get_filepaths()
        new_color_paths, new_depth_paths, new_embedding_paths = [], [], []
        for idx, color_file in enumerate(color_paths):
            frame_id = os.path.basename(color_file).split(".png")[0].split("_")[1]
            c2w = self.get_nearest_pose(frame_id, poses_from_traj, use_interpolation=True, time_distance_threshold=0.2)
            # rotated
            R_z_90 = np.array([
                [0,  1, 0],
                [-1, 0, 0],
                [0,  0, 1],
            ])
            if c2w is not None:
                R_c2w = c2w[:3, :3]
                t_c2w = c2w[:3, 3]
                if self.camera_axis == 'Left':
                    R_c2w_prime = np.dot(R_c2w, R_z_90)
                elif self.camera_axis == 'Up':
                    R_c2w_prime = R_c2w
                adjusted_camera_pose = np.eye(4)
                adjusted_camera_pose[:3, :3] = R_c2w_prime
                adjusted_camera_pose[:3, 3] = t_c2w
                c2w = torch.from_numpy(adjusted_camera_pose).float()
                poses.append(c2w)
                new_color_paths.append(color_file)
                new_depth_paths.append(depth_paths[idx])
                if embedding_paths is not None:
                    new_embedding_paths.append(embedding_paths[idx])
        self.color_paths, self.depth_paths, self.embedding_paths = new_color_paths, new_depth_paths, new_embedding_paths
        return poses
    
    def get_nearest_pose(self, 
                         desired_timestamp,
                         poses_from_traj, 
                         time_distance_threshold = np.inf,
                         use_interpolation = False,
                         interpolation_method = 'split',
                         frame_distance_threshold = np.inf):
        """
        Get the nearest pose to a desired timestamp from a dictionary of poses.

        Args:
            desired_timestamp (float): The timestamp of the desired pose.
            poses_from_traj (dict): A dictionary where keys are timestamps and values are 4x4 transformation matrices representing poses.
            time_distance_threshold (float, optional): The maximum allowable time difference between the desired timestamp and the nearest pose timestamp. Defaults to np.inf.
            use_interpolation (bool, optional): Whether to use interpolation to find the nearest pose. Defaults to False.
            interpolation_method (str, optional): Supports two options, "split" or "geodesic_path". Defaults to "split".

                - "split": performs rigid body motion interpolation in SO(3) x R^3
                - "geodesic_path": performs rigid body motion interpolation in SE(3)
            frame_distance_threshold (float, optional): The maximum allowable distance in terms of frame difference between the desired timestamp and the nearest pose timestamp. Defaults to np.inf.

        Returns:
            (Union[numpy.ndarray, None]): The nearest pose as a 4x4 transformation matrix if found within the specified thresholds, else None.

        Raises:
            ValueError: If an unsupported interpolation method is specified.

        Note:
            If `use_interpolation` is True, the function will perform rigid body motion interpolation between two nearest poses to estimate the desired pose. 
            The thresholds `time_distance_threshold` and `frame_distance_threshold` are used to control how tolerant the function is towards deviations in time and frame distance.
        """

        max_pose_timestamp = max(float(key) for key in poses_from_traj.keys())
        min_pose_timestamp = min(float(key) for key in poses_from_traj.keys()) 

        if float(desired_timestamp) < min_pose_timestamp or \
            float(desired_timestamp) > max_pose_timestamp:
            print('Out')
            return None

        if desired_timestamp in poses_from_traj.keys():
            H = poses_from_traj[desired_timestamp]
        else:
            if use_interpolation:
                greater_closest_timestamp = min(
                    [x for x in poses_from_traj.keys() if float(x) > float(desired_timestamp) ], 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )
                smaller_closest_timestamp = min(
                    [x for x in poses_from_traj.keys() if float(x) < float(desired_timestamp) ], 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )

                if abs(float(greater_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold or \
                    abs(float(smaller_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                    print("Skipping frame.")
                    return None
                
                H0 = poses_from_traj[smaller_closest_timestamp]
                H1 = poses_from_traj[greater_closest_timestamp]
                H0_t = trans(H0)
                H1_t = trans(H1)

                if np.linalg.norm(H0_t - H1_t) > frame_distance_threshold:
                    print("Skipping frame.")
                    return None

                if interpolation_method == "split":
                    H = rigid_interp_split(
                        float(desired_timestamp), 
                        poses_from_traj[smaller_closest_timestamp], 
                        float(smaller_closest_timestamp), 
                        poses_from_traj[greater_closest_timestamp], 
                        float(greater_closest_timestamp)
                    )
                elif interpolation_method == "geodesic_path":
                    H = rigid_interp_geodesic(
                        float(desired_timestamp), 
                        poses_from_traj[smaller_closest_timestamp], 
                        float(smaller_closest_timestamp), 
                        poses_from_traj[greater_closest_timestamp], 
                        float(greater_closest_timestamp)
                    )
                else:
                    raise ValueError(f"Unknown interpolation method {interpolation_method}")

            else:
                closest_timestamp = min(
                    poses_from_traj.keys(), 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )

                if abs(float(closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                    print("Skipping frame.")
                    return None

                H = poses_from_traj[closest_timestamp]

        desired_pose = H

        assert desired_pose.shape == (4, 4)

        return desired_pose
    
    def TrajStringToMatrix(self, traj_str):
        """ 
        Converts a line from the camera trajectory file into translation and rotation matrices

        Args:
            traj_str (str): A space-delimited file where each line represents a camera pose at a particular timestamp. The file has seven columns:

                - Column 1: timestamp
                - Columns 2-4: rotation (axis-angle representation in radians)
                - Columns 5-7: translation (usually in meters)

        Returns:
            (tuple): Tuple containing:

                - ts (str): Timestamp.
                - Rt (numpy.ndarray): Transformation matrix representing rotation and translation.

        Raises:
            AssertionError: If the input string does not have exactly seven columns.
        """

        tokens = traj_str.split()
        assert len(tokens) == 7
        ts = tokens[0]

        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))

        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)

        return (ts, Rt)

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class FunGraph3DDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        cameras, images = read_model(self.input_folder, ".txt")
        for color_p in self.color_paths:
            for image_id, image in images.items():
                image_name = image.name.split('/')[-1]
                if image_name in color_p:
                    world_to_camera = image.world_to_camera
                    c2w = np.linalg.inv(world_to_camera)
                    c2w = torch.from_numpy(c2w).float()
                    poses.append(c2w)
                    break
        return poses


def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings


def convert_angle_axis_to_matrix3(angle_axis):
    """
    Converts a rotation from angle-axis representation to a 3x3 rotation matrix.

    Args:
        angle_axis (numpy.ndarray): A 3-element array representing the rotation in angle-axis form [angle, axis_x, axis_y, axis_z].

    Returns:
        (numpy.ndarray): A 3x3 rotation matrix representing the same rotation as the input angle-axis.

    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix


def rot(H):
    return H[:3, :3]


def trans(H):
    return H[:3, 3]


def inverse(H):
    H_inv = np.eye(4)
    H_inv[0:3, 0:3] = rot(H).T
    H_inv[0:3, 3] = -rot(H).T @ trans(H)

    return H_inv


def rigid_interp_geodesic(t, H0, t0, H1, t1):
    """
    Performs rigid body motion interpolation in SE(3). See https://www.geometrictools.com/Documentation/InterpolationRigidMotions.pdf.

    Args:
        t (float): desired timestep
        H0 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t0
        t0 (float): timestep corresponding to H0
        H1 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t1
        t1 (float): timestep corresponding to H1

    Returns:
        (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t  
    
    """

    # map t in the interval [0, 1]
    slope = (1.0 - 0.0) / (t1 - t0)
    t_ = 0.0 + slope * (t - t0)

    return GeodesicPath(t_, H0, H1)


def rigid_interp_split(t, H0, t0, H1, t1):
    """
    Performs rigid body motion interpolation in SO(3) x R^3. See https://www.adrian-haarbach.de/interpolation-methods/doc/haarbach2018survey.pdf.

    Args:
        t (float): desired timestep
        H0 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t0
        t0 (float): timestep corresponding to H0
        H1 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t1
        t1 (float): timestep corresponding to H1

    Returns:
        (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t  
    
    """

    # map t in the interval [0, 1]
    slope = (1.0 - 0.0) / (t1 - t0)
    t_ = 0.0 + slope * (t - t0)

    H0_R = H0[0:3, 0:3]
    H0_T = H0[0:3, 3]
    H0_new = np.eye(4)
    H0_new[0:3, 0:3] = H0_R

    H1_R = H1[0:3, 0:3]
    H1_T = H1[0:3, 3]
    H1_new = np.eye(4)
    H1_new[0:3, 0:3] = H1_R

    interpH = np.eye(4)

    interpH[0:3, 0:3] = GeodesicPath(t_, H0_new, H1_new)[0:3, 0:3]

    interpH[0:3, 3] = H0_T + t_ * (H1_T - H0_T)

    return interpH


def GeodesicPath(t, H0, H1):

    # If you plan on calling Geodesic Path for the same H0 and H1 but for multiple
    # t−values, the following terms can be precomputed and cached for use by the
    # last block of code
    H = H1 @ InverseRigid(H0)
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    S = Log(H_R)

    s0 = S[2, 1]
    s1 = S[0, 2]
    s2 = S[1, 0]
    theta = np.sqrt(s0*s0 + s1*s1 + s2*s2)
    invV1 = computeInverseV1(theta, S)
    U = invV1 @ H_T
    #### until here the terms can be precomputed for multiple t-values

    interpR = Exp(t, theta, S)
    interpTTimesV = computeTTimesV(t, theta, S)

    interpH = np.eye(4)
    H0_R = H0[0:3, 0:3]
    H0_T = H0[0:3, 3]
    interpH[0:3, 0:3] = interpR @ H0_R
    interpH[0:3, 3] = interpR @ H0_T + interpTTimesV @ U

    return interpH


def InverseRigid(H):
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    invH = np.eye(4)
    invH[0:3, 0:3] = H_R.T
    invH[0:3, 3] = -H_R.T @ H_T

    return invH


def Exp(t, theta, S):
    angle = t * theta
    thetaSqr = theta * theta

    if theta > 0:
        return np.eye(3) + (np.sin(angle) / theta) * S + ((1 - np.cos(angle)) / thetaSqr) * S @ S
    else:
        return np.eye(3)


def Log(R):
    S = np.array((3, 3))

    arg = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1) # in [-1,1]

    if arg > -1:
        if arg < 1:
            # 0 < angle < pi
            angle = np.arccos(arg)
            sinAngle = np.sin(angle)

            c = 0.5 * angle / sinAngle
            S = c * (R - R.T)
        else: # arg = 1, angle = 0
            # R is the identity matrix and S is the zero matrix
            S = np.zeros((3, 3))
    else: # arg = -1, angle = pi
        # Knowing R+I is symmetric and wanting to avoid bias, we use
        # ( R(i,j) + R(j,i) ) / 2 for the off−diagonal entries rather than R(i,j)
        s = np.zeros((3, 1))
        if R[0, 0] >= R[1, 1]:
            if R[0, 0] >= R[2, 2]:
                # r00 is the maximum diagonal term
                s[0] = R[0, 0] + 1
                s[1] = 0.5 * (R[0, 1] + R[1, 0])
                s[2] = 0.5 * (R[0, 2] + R[2, 0])
            else:
                # r22 is the maximum diagonal term
                s[0] = 0.5 * (R[2, 0] + R[0, 2])
                s[1] = 0.5 * (R[2, 1] + R[1, 2])
                s[2] = R[2, 2] + 1
        else:
            if R[1, 1] >= R[2, 2]:
                # r11 is the maximum diagonal term
                s[0] = 0.5 * (R[1, 0] + R[0, 1])
                s[1] = R[1, 1] + 1
                s[2] = 0.5 * (R[1, 2] + R[2, 1])

            else:
                # r22 is the maximum diagonal term
                s[0] = 0.5 * (R[2, 0] + R[0, 2])
                s[1] = 0.5 * (R[2, 1] + R[1, 2])
                s[2] = R[2, 2] + 1

        length = np.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])

        if length > 0:
            adjust = np.pi * np.sqrt(0.5) / length
            s = adjust * s

        else:
            s = np.zeros((3, 1))


        S[0, 0] = 0.0
        S[0, 1] = -s[2]
        s[0, 2] = s[1]
        S[1, 0] = s[2]
        S[1, 1] = 0.0
        S[1, 2] = -s[0]
        S[2, 0] = -s[1]
        S[2, 1] = s[0]
        S[2, 2] = 0.0

    assert S.shape == (3, 3)
    return S


def computeTTimesV(t, theta, S):
    if theta > 0:
        angle = t * theta
        thetaSqr = theta * theta 
        thetaCub = theta * thetaSqr

        c0 = (1 - np.cos(angle)) / thetaSqr
        c1 = (angle - np.sin(angle)) / thetaCub

        return t * np.eye(3) + c0 * S + c1 * S @ S
    else:
        return t * np.eye(3)


def computeInverseV1(theta, S):
    if theta > 0:
        thetaSqr = theta * theta
        c = (1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta)))) / thetaSqr

        return np.eye(3) - 0.5 * S + c * S @ S
    else:
        return np.eye(3)


@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict['dataset_name'].lower() in ['scenefun3d']:
        return SceneFun3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['fungraph3d']:
        return FunGraph3DDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
