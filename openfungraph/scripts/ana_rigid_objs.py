import copy
import pickle
import gzip
import argparse
import numpy as np
import open3d as o3d

from openfungraph.slam.slam_classes import MapObjectList


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--part_result_path", type=str, required=True)
    
    return parser


def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_name] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors


def compute_overlap_ratio(source, target, distance_threshold=0.02):
    # source: part
    # target: object

    # source_tree = o3d.geometry.KDTreeFlann(source)
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    overlap_count = 0
    for point in source.points:
        [_, idx, _] = target_tree.search_radius_vector_3d(point, distance_threshold)
        if len(idx) > 0:
            overlap_count += 1
    
    overlap_ratio = overlap_count / len(source.points)
    return overlap_ratio


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_path = args.result_path
    part_result_path = args.part_result_path

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    with gzip.open(part_result_path, "rb") as fp:
        part_results = pickle.load(fp)
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])

    parts = MapObjectList()
    parts.load_serializable(part_results['objects'])
    
    # Run the post-processing filtering and merging in instructed to do so
    cfg = copy.deepcopy(results['cfg'])

    parts_interest = ["knob", "button", "handle"]

    rigid_inter_id_candidate = []
    part_inter_id_candidate = []

    for inter_idx, obj_inter in enumerate(objects):
        obj_inter['connected_parts'] = []
    
    for inter_idx, obj_inter in enumerate(objects):
        obj_classes_inter = np.asarray(obj_inter['class_name'])
        values_inter, counts_inter = np.unique(obj_classes_inter, return_counts=True)
        obj_class_inter = values_inter[np.argmax(counts_inter)]
        tag = False
        for obj_idx, obj in enumerate(parts):
            obj_classes = np.asarray(obj['class_name'])
            values, counts = np.unique(obj_classes, return_counts=True)
            obj_class = values[np.argmax(counts)]
            if obj_class in parts_interest:
                # an interactable part
                # detect nearby objects of interest
                points_part = obj['pcd']
                points_obj_inter = obj_inter['pcd']
                iou = compute_overlap_ratio(points_part, points_obj_inter, 0.02)
                # fusion based on inter objects: 1 object many parts and object must be big enough
                obj_box_extent = obj_inter['bbox'].extent
                part_box_extent = obj['bbox'].extent
                if iou > 0.7 and obj_box_extent.mean() > 3 * part_box_extent.mean():
                    print(obj_class_inter, ' ', obj_class, ' ', iou)
                    if 'connected_parts' not in obj_inter:
                        # obj_inter['ori_id'] = inter_idx
                        obj_inter['connected_parts'] = []
                        obj_inter['connected_parts'].append(obj_idx)
                        part_inter_id_candidate.append(obj_idx)
                        tag = True
                    else:
                        obj_inter['connected_parts'].append(obj_idx)
                        part_inter_id_candidate.append(obj_idx)
                        tag = True
        if tag:
            rigid_inter_id_candidate.append(inter_idx)

    updated_results = {
        'objects': objects.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
        'inter_id_candidate': rigid_inter_id_candidate
    }    

    save_path = result_path
    
    with gzip.open(save_path, "wb") as f:
        pickle.dump(updated_results, f)
    print(f"Saved full point cloud to {save_path}")

    updated_results = {
        'objects': parts.to_serializable(),
        'cfg': part_results['cfg'],
        'class_names': part_results['class_names'],
        'class_colors': part_results['class_colors'],
        'part_inter_id_candidate': part_inter_id_candidate
    }    

    save_path = part_result_path
    
    with gzip.open(save_path, "wb") as f:
        pickle.dump(updated_results, f)
    print(f"Saved full point cloud to {save_path}")       