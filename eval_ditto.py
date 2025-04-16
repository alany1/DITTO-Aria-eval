import os
import pickle

import numpy as np
import torch
import trimesh
from params_proto import PrefixProto

from aria_utils import evaluate_revolute_joint, evaluate_prismatic_joint, sample_mesh_surface
from aria_utils import normalize as normalize_np
from src.models.geo_art_model_v0 import add_r_joint_to_scene
from src.models.modules.losses_dense_joint import normalize
from src.utils.joint_estimation import aggregate_dense_prediction_r
from src.utils.misc import sample_point_cloud
from tqdm import tqdm
device = 'cuda'

class DittoEvalArgs(PrefixProto):
    data_root = "/home/exx/datasets"
    data_prefix = "aria/blender_eval/kitchen_cgtrader_4449901"

    eval_root = "/home/exx/datasets/aria_experiments"
    eval_prefix = "debug/v0"
    
    mesh_sample_n_points = 10_000


def setup():
    import torch
    from hydra.experimental import (
        initialize,
        compose,
    )
    import hydra

    from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D

    with initialize(config_path='configs/'):
        config = compose(
            config_name='config',
            overrides=[
                'experiment=Ditto_s2m.yaml',
            ], return_hydra_config=True)
    config.datamodule.opt.train.data_dir = './data/'
    config.datamodule.opt.val.data_dir = './data/'
    config.datamodule.opt.test.data_dir = './data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load('./data/Ditto_s2m.ckpt')
    device = torch.device(0)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.eval().to(device)

    generator = Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type='pointcloud',
        refinement_step=0,
        padding=0.1,
        resolution0=32
    )
    
    return model, generator

def make_viz(mesh_dict, joint_axis_pred, pivot_point_pred):
    scene = trimesh.Scene()
    static_part = mesh_dict[0].copy()
    mobile_part = mesh_dict[1].copy()
    scene.add_geometry(static_part)
    scene.add_geometry(mobile_part)
    add_r_joint_to_scene(scene, joint_axis_pred, pivot_point_pred, 1.0, recenter=True)
    
    return scene

def run_one(*, pc_start: np.ndarray, pc_end: np.ndarray, model, generator, override_articulation_type=None):
    bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
    bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
    norm_center = (bound_max + bound_min) / 2
    norm_scale = (bound_max - bound_min).max() * 1.1
    pc_start = (pc_start - norm_center) / norm_scale
    pc_end = (pc_end - norm_center) / norm_scale

    pc_start, _ = sample_point_cloud(pc_start, 4 * 8192)
    pc_end, _ = sample_point_cloud(pc_end, 4 * 8192)
    sample = {
        "pc_start": torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
        "pc_end": torch.from_numpy(pc_end).unsqueeze(0).to(device).float(),
    }

    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = (
            model.model.decode_joints(mobile_points_all, c)
        )

    mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 255], dtype=np.uint8)
    joint_type_prob_og = joint_type_logits.sigmoid().mean()
    joint_type_prob = joint_type_logits.sigmoid().mean()
    
    if override_articulation_type is not None:
        if override_articulation_type == "revolute":
            joint_type_prob = torch.tensor(0.0)
        elif override_articulation_type == "prismatic":
            joint_type_prob = torch.tensor(1.0)
    
    if joint_type_prob.item() < 0.5:
        # axis voting
        joint_r_axis = normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
        p_seg = mobile_points_all[0].cpu().numpy()

        pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
    # prismatic
    else:
        # axis voting
        joint_p_axis = normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()

        pivot_point_pred = mesh_dict[1].bounds.mean(0)
    
    return mesh_dict, joint_axis_pred, pivot_point_pred, joint_type_prob_og
    
    
def main():
    model, generator = setup()
    z = np.load("/home/exx/mit/Ditto/data/Shape2Motion/cabinet_test_standard/scenes/c319659859dd406196e336bfb53282da.npz")
    pc_start = z["pc_start"]
    pc_end = z["pc_end"]
    
    mesh_dict, joint_axis_pred, pivot_point_pred, _ = run_one(
        pc_start=pc_start,
        pc_end=pc_end,
        model=model,
        generator=generator,
    )
    
    scene = make_viz(mesh_dict, joint_axis_pred, pivot_point_pred)
    
    scene.export("/home/exx/Downloads/scene.ply")

def entrypoint(**deps):
    DittoEvalArgs._update(**deps)

    with open(f"{DittoEvalArgs.data_root}/{DittoEvalArgs.data_prefix}/bone_params.pkl", "rb") as f:
        bone_params = pickle.load(f)

    with open(
        f"{DittoEvalArgs.data_root}/{DittoEvalArgs.data_prefix}/articulated_objects.pkl", "rb"
    ) as f:
        articulated_objects = pickle.load(f)
        
    metrics = dict()
    model, generator = setup()


    for obj, articulation_type in tqdm(articulated_objects.items(), desc="Evaluating ditto..."):
        armature, bone = obj.split(":")
        params = bone_params[armature][bone]
        assert articulation_type in ["revolute", "prismatic"]
        
        mesh_start_path = f"{DittoEvalArgs.data_root}/{DittoEvalArgs.data_prefix}/ditto/{obj}/mesh_start.ply"
        mesh_end_path = f"{DittoEvalArgs.data_root}/{DittoEvalArgs.data_prefix}/ditto/{obj}/mesh_end.ply"
        
        start_pcd_o3d = sample_mesh_surface(mesh_start_path, num_points=DittoEvalArgs.mesh_sample_n_points)
        end_pcd_o3d = sample_mesh_surface(mesh_end_path, num_points=DittoEvalArgs.mesh_sample_n_points)
        
        start_pcd = np.asarray(start_pcd_o3d.points)
        end_pcd = np.asarray(end_pcd_o3d.points)

        bound_max = np.maximum(start_pcd.max(0), end_pcd.max(0))
        bound_min = np.minimum(start_pcd.min(0), end_pcd.min(0))
        norm_center = (bound_max + bound_min) / 2
        norm_scale = (bound_max - bound_min).max() * 1.1

        mesh_dict, joint_axis_pred, pivot_point_pred, p_pris = run_one(pc_start=start_pcd, 
                    pc_end=end_pcd,
                    model=model,
                    generator=generator,
                    override_articulation_type=articulation_type)        
        
        est_axis_dir = joint_axis_pred
        est_pivot = pivot_point_pred * norm_scale + norm_center
        
        if articulation_type == "revolute":
            gt_axis_dir = np.array(params["axis_dir"])
            gt_pivot = np.array(params["position"])
            
            
            axis_error, pivot_error = evaluate_revolute_joint(
                est_axis=normalize_np(est_axis_dir),
                est_pivot=est_pivot,
                gt_axis=normalize_np(gt_axis_dir),
                gt_pivot=gt_pivot,
            )
            pivot_error /= params["characteristic_length"]
            
            metrics_dict = {
                "articulation_type": articulation_type,
                "correct": True,
                "error": {
                    "axis_error": axis_error,
                    "pivot_error": pivot_error,
                },
            }

        else:
            gt_axis_dir = np.array(params["axis_dir"])

            axis_error = evaluate_prismatic_joint(
                est_axis=normalize_np(est_axis_dir),
                gt_axis=normalize_np(gt_axis_dir),
            )

            metrics_dict = {
                "articulation_type": articulation_type,
                "correct": True,
                "error": {
                    "axis_error": axis_error,
                },
            }

        metrics[obj] = metrics_dict

    # log metrics
    save_path = (
        f"{DittoEvalArgs.eval_root}/{DittoEvalArgs.eval_prefix}/{DittoEvalArgs.data_prefix}/ditto/metrics.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"Logged metrics to {save_path}!")

    from pprint import pprint

    pprint(metrics)


if __name__ == '__main__':
    entrypoint()
