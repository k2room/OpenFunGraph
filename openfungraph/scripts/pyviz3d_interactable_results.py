import copy
import os
import pickle
import gzip
import argparse
import numpy as np
import open3d as o3d
from openfungraph.slam.slam_classes import MapObjectList
import pyviz3d.visualizer as viz


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter_result_path", type=str, default=None)
    parser.add_argument("--part_result_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--pc_path", type=str, default=None)
    parser.add_argument("--pose_path", type=str, default=None)
    return parser


def load_result(result_path):

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])

    if 'bg_objects' not in results:
        bg_objects = None
    elif results['bg_objects'] is None:
        bg_objects = None
    else:
        bg_objects = MapObjectList()
        bg_objects.load_serializable(results["bg_objects"])

    class_colors = results['class_colors']
    class_names = results['class_names']
    try:
        obj_cand = results['inter_id_candidate']
    except:
        obj_cand = results['part_inter_id_candidate']

    return objects, bg_objects, class_colors, class_names, obj_cand


def main(args):
    inter_result_path = args.inter_result_path
    part_result_path = args.part_result_path
        
    objects, _, class_colors, class_names, obj_cand = load_result(inter_result_path)
    parts, _, part_colors, _, part_cand = load_result(part_result_path)

    if args.pc_path is not None:
        scene_pc = o3d.io.read_point_cloud(args.pc_path)
        if args.pose_path is not None:
            scene_pc.transform(np.load(args.pose_path))
        pc_center = np.mean(np.asarray(scene_pc.points), axis=0)
    else:
        pc_center = np.array([0.0,0.0,0.0])
    
    if args.edge_file is not None:
        with open(args.edge_file, "rb") as f:
            edges = pickle.load(f)
    
    v = viz.Visualizer()

    normals = []
    obj_centers = []
    # Sub-sample the point cloud for better interactive experience
    for i in obj_cand:
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(0.01)
        objects[i]['pcd'] = pcd
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals.append(np.asarray(pcd.normals))
        obj_centers.append(np.mean(np.asarray(pcd.points), axis=0)-pc_center)
    
    for i in part_cand:
        pcd = parts[i]['pcd']
        pcd = pcd.voxel_down_sample(0.01)
        parts[i]['pcd'] = pcd
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals.append(np.asarray(pcd.normals))
        obj_centers.append(np.mean(np.asarray(pcd.points), axis=0)-pc_center)
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    part_pcds = copy.deepcopy(parts.get_values("pcd"))

    # Get the color for each object when colored by their class
    object_classes = []
    object_class_names = []
    object_colors = []
    obj_color_dict = {}
    points = []
    colors = []
    for i in obj_cand:
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
        object_class_names.append(obj['refined_obj_tag'])
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0
        object_colors.append(np.array([r,g,b]))
        obj_color_dict['O' + str(i)] = np.array([r,g,b])
        points.append(np.asarray(pcd.points)-pc_center)
        colors.append(np.asarray(pcd.colors))
    
    part_color_dict = {}
    for i in part_cand:
        obj = parts[i]
        pcd = part_pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
        object_class_names.append(obj['refined_obj_tag'])
        if 'knob' in obj['refined_obj_tag']:
            if 'knob' not in part_color_dict.keys():
                r = np.random.randint(0, 256)/255.0
                g = np.random.randint(0, 256)/255.0
                b = np.random.randint(0, 256)/255.0
                part_color_dict['knob'] = np.array([r,g,b])
            object_colors.append(part_color_dict['knob'])
            obj_color_dict['P' + str(i)] = part_color_dict['knob']
        elif 'handle' in obj['refined_obj_tag']:
            if 'handle' not in part_color_dict.keys():
                r = np.random.randint(0, 256)/255.0
                g = np.random.randint(0, 256)/255.0
                b = np.random.randint(0, 256)/255.0
                part_color_dict['handle'] = np.array([r,g,b])
            object_colors.append(part_color_dict['handle'])
            obj_color_dict['P' + str(i)] = part_color_dict['handle']
        elif 'button' in obj['refined_obj_tag']:
            if 'button' not in part_color_dict.keys():
                r = np.random.randint(0, 256)/255.0
                g = np.random.randint(0, 256)/255.0
                b = np.random.randint(0, 256)/255.0
                part_color_dict['button'] = np.array([r,g,b])
            object_colors.append(part_color_dict['button'])
            obj_color_dict['P' + str(i)] = part_color_dict['button']
        else:
            r = np.random.randint(0, 256)/255.0
            g = np.random.randint(0, 256)/255.0
            b = np.random.randint(0, 256)/255.0
            object_colors.append(np.array([r,g,b]))
            obj_color_dict['P' + str(i)] = np.array([r,g,b])
        points.append(np.asarray(pcd.points)-pc_center)
        colors.append(np.asarray(pcd.colors))
    
    for idx, point in enumerate(points):
        if idx < len(obj_cand):
            v.add_points(f'O{obj_cand[idx]}', point, colors[idx] * 255, normals[idx], point_size=20, visible=True)
            v.add_points(f'InsO{obj_cand[idx]}', np.expand_dims(obj_centers[idx], axis=0), np.expand_dims(object_colors[idx], axis=0) * 255, point_size=200, resolution=15, visible=True)
        else:
            v.add_points(f'P{part_cand[idx - len(obj_cand)]}', point, colors[idx] * 255, normals[idx], point_size=20, visible=True)
            v.add_points(f'InsP{part_cand[idx - len(obj_cand)]}', np.expand_dims(obj_centers[idx], axis=0), np.expand_dims(object_colors[idx], axis=0) * 255, point_size=150, resolution=15, visible=True)
    # add whole pc
    if args.pc_path is not None:
        scene_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        scene_normals = np.asarray(scene_pc.normals)
        v.add_points('Background', np.asarray(scene_pc.points) - pc_center, np.asarray(scene_pc.colors)*255, scene_normals, point_size=20, visible=True)

    for idx in range(len(object_class_names)):
        if idx < len(obj_cand):
            v.add_labels(f'LabelO{obj_cand[idx]}',
                    'O' + str(obj_cand[idx]) + ': ' + object_class_names[idx],
                    # ['O' + str(obj_cand[idx])],
                    obj_centers[idx],
                    object_colors[idx] * 255,
                    '11px',
                    visible=True)
        else:
            v.add_labels(f'LabelP{part_cand[idx - len(obj_cand)]}',
                    'P' + str(part_cand[idx - len(obj_cand)]) + ': ' + object_class_names[idx],
                    # ['P' + str(part_cand[idx - len(obj_cand)])],
                    obj_centers[idx],
                    object_colors[idx] * 255,
                    '11px',
                    visible=True)
    
    all_obj_centers = []
    all_part_centers = []
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        all_obj_centers.append(np.mean(np.asarray(pcd.points), axis=0)-pc_center)
    for i in range(len(parts)):
        pcd = parts[i]['pcd']
        all_part_centers.append(np.mean(np.asarray(pcd.points), axis=0)-pc_center)

    if args.edge_file is not None:
        edge_func = []
        edge_centers = []
        for edge in edges:
            # [(0:obj_idx: obj.pkl, 1:part_idx: part.pkl，2:-1，3:description)] rigid
            # [(0:obj_idx: obj.pkl, 1:-1，2:obj_idx: obj.pkl，3:description)] remote
            if edge[2] == -1:
                edge_func.append(edge[3])
                edge_centers.append((all_obj_centers[edge[0]] + all_part_centers[edge[1]]) / 2)
                v.add_polyline(f'EdgeOP{edge[0]}{edge[1]}', np.array([all_obj_centers[edge[0]], all_part_centers[edge[1]]]), color=np.array([100.0, 100.0, 100.0]), edge_width=0.02, alpha=0.5)
            elif edge[1] == -1:
                edge_func.append(edge[3])
                edge_centers.append((all_obj_centers[edge[0]] + all_obj_centers[edge[2]]) / 2)
                v.add_polyline(f'EdgeOO{edge[0]}{edge[2]}', np.array([all_obj_centers[edge[0]], all_obj_centers[edge[2]]]), color=np.array([100.0, 100.0, 100.0]), edge_width=0.02, alpha=0.5)
        edge_centers = np.stack(edge_centers)
        
    if args.edge_file is not None:
        for i in range(len(edge_func)):
            v.add_labels(f'EdgeLabels{i}',
                        edge_func[i],
                        edge_centers[i],
                        np.ones(3) * 255,
                        '8px',
                        visible=True)

    v.save(os.path.dirname(inter_result_path)+ '/fungraph')
    np.save(os.path.dirname(inter_result_path)+ '/fungraph/color.npy', obj_color_dict)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)