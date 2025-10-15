import json
import open3d as o3d
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BertModel, BertTokenizer
import numpy as np
import argparse
import pickle
import gzip
from openfungraph.slam.slam_classes import MapObjectList
from tqdm import tqdm
from openfungraph.utils.ious import compute_3d_iou


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.)

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


def compute_iou(pc1, pc2, voxel_size):
    # Voxelization
    pcd1 = pc1.voxel_down_sample(voxel_size)
    pcd2 = pc2.voxel_down_sample(voxel_size)

    # Create binary occupancy grids
    min_bound = np.minimum(pcd1.get_min_bound(), pcd2.get_min_bound())
    max_bound = np.maximum(pcd1.get_max_bound(), pcd2.get_max_bound())
    
    # Define grid size
    grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Create occupancy grid
    grid1 = np.zeros(grid_size, dtype=bool)
    grid2 = np.zeros(grid_size, dtype=bool)

    # Fill occupancy grids
    for point in np.asarray(pcd1.points):
        grid_idx = np.floor((point - min_bound) / voxel_size).astype(int)
        grid1[tuple(grid_idx)] = True

    for point in np.asarray(pcd2.points):
        grid_idx = np.floor((point - min_bound) / voxel_size).astype(int)
        grid2[tuple(grid_idx)] = True

    # Calculate intersection and union
    intersection = np.sum(grid1 & grid2)
    union = np.sum(grid1 | grid2)

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == 'SceneFun3D':
        with open(args.root_path+'/SceneFun3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
        with open(args.root_path+'/SceneFun3D.relations.json', 'r') as f:
            gt_edge_annos = json.load(f)
        gt_edge_annos = [anno for anno in gt_edge_annos if anno['scene_id'] == args.scene]
    elif args.dataset == 'FunGraph3D':
        with open(args.root_path+'/FunGraph3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
        with open(args.root_path+'/FunGraph3D.relations.json', 'r') as f:
            gt_edge_annos = json.load(f)
        gt_edge_annos = [anno for anno in gt_edge_annos if anno['scene_id'] == args.scene]
    else:
        exit(1)
    
    if args.dataset == 'SceneFun3D':
        # scene_pc = o3d.io.read_point_cloud(args.root_path+'/scans/'+args.scene+'_laser_scan.ply')
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.scene+'_laser_scan.ply')
        refined_transform = np.load(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/'+args.video+'_refined_transform.npy') 
        scene_pc.transform(refined_transform)
    elif args.dataset == 'FunGraph3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/scans/'+args.scene+'.ply')
    
    if args.dataset == 'SceneFun3D':
        objects, _, _, _, _ = load_result(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz')
        parts, _, _, _, _ = load_result(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz')
        with open(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/cfslam_funcgraph_edges.pkl', "rb") as f:
            edges = pickle.load(f)
    elif args.dataset == 'FunGraph3D':
        objects, _, _, _, _ = load_result(args.root_path+'/'+args.scene+'/'+args.video+'/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz')
        parts, _, _, _, _ = load_result(args.root_path+'/'+args.scene+'/'+args.video+'/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz')
        with open(args.root_path+'/'+args.scene+'/'+args.video+'/cfslam_funcgraph_edges.pkl', "rb") as f:
            edges = pickle.load(f)

    all_labels_embeddings = np.load(args.root_path+'/all_labels_clip_embeddings.npy')
    with open(args.root_path+'/all_labels.json', 'r') as f:
        all_labels = json.load(f)
    
    all_edges_embeddings = np.load(args.root_path+'/all_edges_bert_embeddings.npy')
    with open(args.root_path+'/all_edges.json', 'r') as f:
        all_edges = json.load(f)

    rk_num = 0
    rk_num_obj = 0
    rk_num_edge = 0

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')

    fail_num = 0
    for gt_edge in tqdm(gt_edge_annos):
        try:
            gt_obj1 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["first_node_annot_id"]][0]  # functionality
        except:
            fail_num += 1
            continue
        # start to retrieve functionality
        gt_label1 = gt_obj1['label']
        gt_mask1 = gt_obj1['indices']
        gt_pc1 = np.asarray(scene_pc.points)[gt_mask1]
        gt_pc1_o3d = o3d.geometry.PointCloud()
        gt_pc1_o3d.points = o3d.utility.Vector3dVector(gt_pc1)
        gt_bbd1 = gt_pc1_o3d.get_oriented_bounding_box()
        # retrieve object
        try:
            gt_obj2 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["second_node_annot_id"]][0]  # obj
        except:
            fail_num += 1
            continue
        gt_label2 = gt_obj2['label']
        gt_mask2 = gt_obj2['indices']
        gt_pc2 = np.asarray(scene_pc.points)[gt_mask2]
        gt_pc2_o3d = o3d.geometry.PointCloud()
        gt_pc2_o3d.points = o3d.utility.Vector3dVector(gt_pc2)
        gt_bbd2 = gt_pc2_o3d.get_oriented_bounding_box()
        # retrieve edge
        gt_rel = gt_edge["description"]
        flag_obj = False
        flag_edge = False
        for edge in edges:
            if edge[2] == -1:
                # edge[1]: functionality
                pred_func = parts[edge[1]]
            elif edge[1] == -1:
                # edge[0]: functionality
                pred_func = objects[edge[0]]
            pred_bbd1 = pred_func['bbox']
            pred_label1 = pred_func['refined_obj_tag']
            iou1 = compute_3d_iou(gt_bbd1, pred_bbd1)
            if iou1 > args.iou_threshold:
                inputs = processor(text=[pred_label1], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-args.topk:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label1 not in topk_label:
                    continue
            else:
                continue
            # continue obj retrieval
            if edge[2] == -1:
                # edge[0]: obj
                pred_obj = objects[edge[0]]
            elif edge[1] == -1:
                # edge[2]: obj
                pred_obj = objects[edge[2]]
            pred_bbd2 = pred_obj['bbox']
            pred_label2 = pred_obj['refined_obj_tag']
            iou2 = compute_3d_iou(gt_bbd2, pred_bbd2)
            if iou2 > args.iou_threshold:
                inputs = processor(text=[pred_label2], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-args.topk:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label2 not in topk_label:
                    continue
            else:
                continue
            flag_obj = True
            # continue edge retrieval
            pred_rel = edge[3]
            inputs = tokenizer_bert(pred_rel, return_tensors='pt', padding=True, truncation=True)
            outputs = model_bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            norm_embeddings = embeddings / np.linalg.norm(embeddings)
            norm_all_edges_embeddings = all_edges_embeddings / np.linalg.norm(all_edges_embeddings, axis=1, keepdims=True)
            similarity = np.dot(norm_embeddings, norm_all_edges_embeddings.T)
            topk_indices = np.argsort(similarity[0], axis=0)[-args.topk:][::-1]
            topk_label = [all_edges[idx] for idx in topk_indices]
            if gt_rel in topk_label:
                flag_edge = True
                rk_num += 1
                break
        if flag_obj:
            rk_num_obj += 1
        if flag_edge:
            rk_num_edge += 1
       
    print('Top ', args.topk, ' Recall: ', rk_num, ' / ', (len(gt_edge_annos) - fail_num), ': ', rk_num / (len(gt_edge_annos) - fail_num))
    print('Top ', args.topk, ' Object Recall: ', rk_num_obj, ' / ', (len(gt_edge_annos) - fail_num), ': ', rk_num_obj / (len(gt_edge_annos) - fail_num))
    print('Top ', args.topk, ' Edge Recall: ', rk_num_edge, ' / ', rk_num_obj, ': ', rk_num_edge / rk_num_obj)

    rk_num = 0
    rk_num_obj = 0
    rk_num_edge = 0

    fail_num = 0
    for gt_edge in tqdm(gt_edge_annos):
        try:
            gt_obj1 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["first_node_annot_id"]][0]  # functionality
        except:
            fail_num += 1
            continue
        # start to retrieve functionality
        gt_label1 = gt_obj1['label']
        gt_mask1 = gt_obj1['indices']
        gt_pc1 = np.asarray(scene_pc.points)[gt_mask1]
        gt_pc1_o3d = o3d.geometry.PointCloud()
        gt_pc1_o3d.points = o3d.utility.Vector3dVector(gt_pc1)
        gt_bbd1 = gt_pc1_o3d.get_oriented_bounding_box()
        # retrieve object
        try:
            gt_obj2 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["second_node_annot_id"]][0]  # obj
        except:
            fail_num += 1
            continue
        gt_label2 = gt_obj2['label']
        gt_mask2 = gt_obj2['indices']
        gt_pc2 = np.asarray(scene_pc.points)[gt_mask2]
        gt_pc2_o3d = o3d.geometry.PointCloud()
        gt_pc2_o3d.points = o3d.utility.Vector3dVector(gt_pc2)
        gt_bbd2 = gt_pc2_o3d.get_oriented_bounding_box()
        # retrieve edge
        gt_rel = gt_edge["description"]
        flag_obj = False
        flag_edge = False
        for edge in edges:
            if edge[2] == -1:
                # edge[1]: functionality
                pred_func = parts[edge[1]]
            elif edge[1] == -1:
                # edge[0]: functionality
                pred_func = objects[edge[0]]
            pred_bbd1 = pred_func['bbox']
            pred_label1 = pred_func['refined_obj_tag']
            iou1 = compute_3d_iou(gt_bbd1, pred_bbd1)
            if iou1 > args.iou_threshold:
                inputs = processor(text=[pred_label1], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-10:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label1 not in topk_label:
                    continue
            else:
                continue
            # continue obj retrieval
            if edge[2] == -1:
                # edge[0]: obj
                pred_obj = objects[edge[0]]
            elif edge[1] == -1:
                # edge[2]: obj
                pred_obj = objects[edge[2]]
            pred_bbd2 = pred_obj['bbox']
            pred_label2 = pred_obj['refined_obj_tag']
            iou2 = compute_3d_iou(gt_bbd2, pred_bbd2)
            if iou2 > args.iou_threshold:
                inputs = processor(text=[pred_label2], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-10:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label2 not in topk_label:
                    continue
            else:
                continue
            flag_obj = True
            # continue edge retrieval
            pred_rel = edge[3]
            inputs = tokenizer_bert(pred_rel, return_tensors='pt', padding=True, truncation=True)
            outputs = model_bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            norm_embeddings = embeddings / np.linalg.norm(embeddings)
            norm_all_edges_embeddings = all_edges_embeddings / np.linalg.norm(all_edges_embeddings, axis=1, keepdims=True)
            similarity = np.dot(norm_embeddings, norm_all_edges_embeddings.T)
            topk_indices = np.argsort(similarity[0], axis=0)[-10:][::-1]
            topk_label = [all_edges[idx] for idx in topk_indices]
            if gt_rel in topk_label:
                flag_edge = True
                rk_num += 1
                break
        if flag_obj:
            rk_num_obj += 1
        if flag_edge:
            rk_num_edge += 1
       
    print('Top 10 Recall: ', rk_num, ' / ', (len(gt_edge_annos) - fail_num), ': ', rk_num / (len(gt_edge_annos) - fail_num))
    print('Top 10 Object Recall: ', rk_num_obj, ' / ', (len(gt_edge_annos) - fail_num), ': ', rk_num_obj / (len(gt_edge_annos) - fail_num))
    print('Top 10 Edge Recall: ', rk_num_edge, ' / ', rk_num_obj, ': ', rk_num_edge / rk_num_obj)