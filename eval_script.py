import json

import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

from diffusers import StableDiffusionDepth2ImgPipeline

import open3d as o3d

import sys
import os

from utils.preprocessing import *

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(parent_dir, 'utils'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EvalPipeline:
    prompt_list = ["old purple victorian room", "bright white scientific laboratory", "medieval dungeon",
                   "greenhouse full of plants"]

    prompt = ""

    #nyu_path = "data/nyu_depth_v2_labeled.mat"
    nyu_result_root = "./results/NYU/"

    guidance_scale = 12
    strength = 0.75
    num_inference_steps = 25
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self,
                      #nyu_path = "data/nyu_depth_v2_labeled.mat",
                       nyu_result_root = "./results/NYU/", prompt="", guidance_scale = 12,
                       strength = 0.75, num_inference_steps = 25):

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = self.prompt_list[0]

        #self.nyu_path = nyu_path
        self.nyu_result_root = nyu_result_root

        model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        self.midas.to(device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
        ).to(device)

        self.depth2img_pipe.enable_attention_slicing()

        self.guidance_scale = guidance_scale
        self.strength = strength
        self.num_inference_steps = num_inference_steps

    def infer_depth2img(self, source_np, prompt, guidance_scale=guidance_scale, depth_image=None, strength=strength,
                        num_inference_steps=num_inference_steps, save_name='', comparison_save_name=''):

        source_image = Image.fromarray(source_np)

        if depth_image is None:
            results = self.depth2img_pipe(prompt=prompt, image=source_image, strength=strength,
                                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        else:
            depth_tensor = torch.from_numpy(np.expand_dims(depth_image, axis=0))
            results = self.depth2img_pipe(prompt=prompt, image=source_image, depth_map=depth_tensor, strength=strength,
                                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(source_np)
        ax[1].imshow(results.images[0])
        fig.suptitle(prompt, fontsize=16, y=0.8)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        if save_name:
            results.images[0].save(save_name)
        if comparison_save_name:
            fig.savefig(comparison_save_name)

        plt.show()

        return results.images[0]

    def generate_img(self, image, prompt, depth_image=None, with_depth=True, save_folder="", img_id=""):

        if save_folder:
            gen_img_save_name = save_folder + f"{img_id}_generated.png"
            comparison_save_name = save_folder + f"{img_id}_comparison.png"
        else:
            gen_img_save_name = comparison_save_name = ""

        generated_image = self.infer_depth2img(image, prompt, depth_image=depth_image,
                                          guidance_scale=self.guidance_scale, strength=self.strength,
                                          num_inference_steps=self.num_inference_steps,
                                          save_name=gen_img_save_name, comparison_save_name=comparison_save_name)

        print("generated image--")

        # results = infer_depth2img(source_img, prompt_list[0], depth_image,  guidance_scale, strength, num_inference_steps)
        return generated_image

    def eval(self, src_img_np, ground_depth_map, i):

        print(f"max of rgb: {np.max(src_img_np)}, min: {np.min(src_img_np)}, mean: {np.mean(src_img_np)}")
        print(
            f"max of depth: {np.max(ground_depth_map)}, min: {np.min(ground_depth_map)}, mean: {np.mean(ground_depth_map)}")

        gen_img = self.generate_img(src_img_np, prompt=self.prompt, depth_image=ground_depth_map, with_depth=True,
                           save_folder=self.nyu_result_root, img_id=i)

        gen_img_np = np.array(gen_img)

        # ground_depth_map = infer_depth_map(source_img_np)
        predict_depth_path = self.nyu_result_root + f"{i}_gen_depth.png"
        predict_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)

        predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        heatmap_path = self.nyu_result_root + f"{i}_depth_heatmap.png"
        heatmap = self.get_depth_heat_map(ground_depth_map, predict_depth_map_aligned,
                                     img_id=i, save_name=heatmap_path)

        original_pcd = get_point_could(src_img_np, ground_depth_map)
        generated_pcd = get_point_could(gen_img_np, predict_depth_map_aligned)

        ground_pcd_path = self.nyu_result_root + f"{i}_ground_pcd.png"
        gen_pcd_path = self.nyu_result_root + f"{i}_gen_pcd.png"
        view_setting_path = self.nyu_result_root + "view_setting.json"

        capture_pcd_with_view_params(pcd=original_pcd, pcd_path=ground_pcd_path, view_setting_path=view_setting_path)
        capture_pcd_with_view_params(pcd=generated_pcd, pcd_path=gen_pcd_path, view_setting_path=view_setting_path)




def get_point_could(rgb_image, depth_image):
    new_depth_image = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image), o3d.geometry.Image(new_depth_image),
        depth_scale=255, depth_trunc=255.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0104, max_nn=12))

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

    return pcd


def get_depth_heat_map(ground_depth_map, predict_depth_map, img_id=None, save_name=''):
    depth_diff = ground_depth_map - predict_depth_map

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), layout='constrained')
    ax[0].imshow(ground_depth_map)
    ax[1].imshow(predict_depth_map)
    diff = ax[2].imshow(depth_diff, cmap='RdBu_r')

    cbar = fig.colorbar(diff, ax=ax[2], shrink=0.6)
    cbar.set_label('ground truth - predicted', rotation=90, labelpad=5)
    cbar.ax.set_yticklabels(["{:.0%}".format(i / 255) for i in cbar.get_ticks()])  # set ticks of your format

    ax[0].set_title('Ground Truth', fontsize=16)
    ax[1].set_title('Generated', fontsize=16)
    ax[2].set_title('Difference Heat Map', fontsize=16)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    if img_id is not None:
        fig.suptitle(f"Depth Maps for Image - {img_id}", fontsize=18, y=0.95)
    else:
        fig.suptitle(f"Depth Maps", fontsize=18, y=0.95)

    if save_name:
        fig.savefig(save_name)

    plt.show()


def get_mesh_from_pcd(pcd, method="poisson", visualise=False):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

        if method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                                        depth=10, width=0, scale=1.1,
                                                                                        linear_fit=True)

            if visualise:
                o3d.visualization.draw_geometries([mesh])

            return mesh, densities

        if method == "ball_pivoting":
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            if visualise:
                o3d.visualization.draw_geometries([pcd, mesh])

            return mesh

        if method == "alpha":
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

            for alpha in np.logspace(np.log10(0.5), np.log10(0.1), num=4):
                print(f"alpha={alpha:.3f}")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map)
                mesh.compute_vertex_normals()
                if visualise:
                    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

            return mesh


def capture_pcd_with_view_params(pcd, pcd_path, view_setting_path):
    vis = o3d.visualization.Visualizer()

    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    with open(view_setting_path, "r") as f:
        js = json.load(f)

    vc = vis.get_view_control()
    vc.change_field_of_view(js['trajectory'][0]['field_of_view'])
    vc.set_front(js['trajectory'][0]['front'])
    vc.set_lookat(js['trajectory'][0]['lookat'])
    vc.set_up(js['trajectory'][0]['up'])
    vc.set_zoom(js['trajectory'][0]['zoom'])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(pcd_path)
    vis.destroy_window()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--nyu_path', default='/data/nyu_depth_v2_labeled.mat')
    parser.add_argument('--nyu_result_root', default='./results/NYU/')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--guidance_scale', default=12, type=int)
    parser.add_argument('--strength', default=0.75, type=float)
    parser.add_argument('--num_inference_steps', type=int, default=25)

    args = parser.parse_args()

    evalPipe = EvalPipeline(
                       nyu_result_root = args.nyu_result_root, prompt=args.prompt, guidance_scale = args.guidance_scale,
                       strength = args.strength, num_inference_steps = args.num_inference_steps)

    nyu_path = nyu_path = args.nyu_path
    #eval_results_path = nyu_result_root + "eval_logs.csv"

    # read mat file
    f = h5py.File(nyu_path)

    rgb_images = f['images']
    depth_maps = f['depths']

    for i in range(0, rgb_images.shape[0]):
        src_img_np, ground_depth_map = prepare_nyu_data(rgb_images[i], depth_maps[i])

        evalPipe.eval(src_img_np, ground_depth_map)

if __name__ == "__main__":
    main()


# python eval_script.py --prompt "scenic jungle temple" --guidance_scale 15 --num_inference_steps 35
#
#