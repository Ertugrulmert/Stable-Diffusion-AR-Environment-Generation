import argparse, sys, os, time
from random import randrange
import gc

from diffusers import (ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from utils.evaluation import Evaluator
from utils.preprocessing import *
from models.model_data import *
from utils.visualisation import Visualiser as vis
from models_3d import point_clouds, mesh_processing

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import random

random.seed(1)

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"---- Device found: {device} ----")

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # "SG161222/Realistic_Vision_V2.0"

# Defaults ------------
default_resolution = 512
default_num_steps = 20
default_conditioning_scale = [1.0, 0.7]
default_guidance_scale = 7.5
seed = 1


class ControlNetModelWrapper:

    def __init__(self, condition_type="depth", multi_condition=False, only_ground=False,  # remote_inference=False,
                 depth_model_type="MiDaS",
                 result_root='./results/NYU/',
                 resolution=256,
                 num_steps=30,
                 cache_dir=""):
        self.result_root = result_root
        self.multi_condition = multi_condition
        self.max_resolution = resolution
        self.resolution = resolution
        self.num_steps = num_steps
        # self.remote_inference = remote_inference
        self.load_depth_model(model_type=depth_model_type, cache_dir=cache_dir)
        if not only_ground:
            self.load_condition_model(condition_type=condition_type, cache_dir=cache_dir)
            self.load_controlnet(cache_dir=cache_dir)
            self.evaluator = Evaluator(condition_type=self.condition_type, multi_condition=multi_condition,
                                       cache_dir=cache_dir)

    # Model Loading

    def load_depth_model(self, model_type="MiDaS", cache_dir=""):
        # Loading Depth Model
        self.depth_model_type = model_type

        if cache_dir:
            torch.hub.set_dir(cache_dir + "/torch")

        if self.depth_model_type == "MiDaS":
            self.depth_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
            # self.depth_model.to(device)
            self.depth_model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.depth_model_type == "DPT_Large" or self.depth_model_type == "DPT_Hybrid":
            self.depth_transform = midas_transforms.dpt_transform
        elif self.depth_model_type == "MiDaS":
            self.depth_transform = midas_transforms.default_transform
        else:
            self.depth_transform = midas_transforms.small_transform

    def load_condition_model(self, condition_type="seg", cache_dir=""):

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        if self.multi_condition:
            self.condition_type = ["seg", "depth"]
        else:
            self.condition_type = condition_type

        if self.condition_type == "seg" or self.multi_condition:
            self.image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade",
                                                                            **kwargs)
            self.segmentation_model = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-large-ade", **kwargs)

    def load_controlnet(self, cache_dir=""):

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        torch.cuda.empty_cache()
        if self.multi_condition:
            self.controlnet_model_id = [ModelData.CONTROLNET_MODEL_IDS[type] for type in self.condition_type]
            controlnet = [ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16, **kwargs)
                          for model_id in self.controlnet_model_id]
        else:
            self.controlnet_model_id = ModelData.CONTROLNET_MODEL_IDS[self.condition_type]
            controlnet = ControlNetModel.from_pretrained(self.controlnet_model_id, torch_dtype=torch.float16,
                                                         **kwargs)

        self.coltrolnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            safety_checker=None,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            **kwargs)

        self.coltrolnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.coltrolnet_pipe.scheduler.config)
        # the library versions sued for this implementation did nto allow for the following optimizations,
        # if possible they must be used

        # self.coltrolnet_pipe.enable_xformers_memory_efficient_attention()
        # self.coltrolnet_pipe.enable_attention_slicing(1)
        # self.coltrolnet_pipe.enable_vae_tiling()
        # self.coltrolnet_pipe.enable_sequential_cpu_offload()
        self.coltrolnet_pipe.to(device)
        torch.cuda.empty_cache()
        gc.collect()

    def infer_depth_map(self, image, save_name='', display=True):

        input_batch = self.depth_transform(image)  # .to(device)

        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            output = prediction.cpu().numpy()

            if display:
                plt.imshow(output)
                plt.show()

            if save_name:
                with open(save_name + '.npy', 'wb') as f:
                    np.save(f, output, allow_pickle=True, fix_imports=True)

            return output

    # inferring the segmentation map
    def infer_seg_ade20k(self, image, i, save_name='', scaling_factor=None, display=False):

        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)

        H, W = image.shape[:2]
        if scaling_factor is not None:
            H *= scaling_factor
            W *= scaling_factor
            H = int(np.round(H / 64.0)) * 64
            W = int(np.round(W / 64.0)) * 64

        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

        for label, color in enumerate(ModelData.ade_palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)
        control_image = Image.fromarray(color_seg)

        if save_name:
            control_image.save(save_name + ".png")

        if display:
            fontsize = 12
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[1].imshow(control_image)
            fig.suptitle(f"Predicted ADE20k Segmentation for Image {i}", fontsize=fontsize, y=0.9)
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            fig.tight_layout()
            plt.show()

        return control_image, H, W

    def infer_controlnet(self, source_image, prompt, H, W, guidance_scale=7.5, conditioning_scale=1.0,
                         condition_image=None, num_inference_steps=10, save_name='', comparison_save_name='',
                         display=False):

        generator = torch.Generator().manual_seed(seed)

        prompt = f'{prompt}, {ModelData.additional_prompt}'

        with torch.inference_mode():
            results = self.coltrolnet_pipe(prompt=prompt, negative_prompt=ModelData.negative_prompt,
                                           image=condition_image,
                                           height=H,
                                           width=W,
                                           guidance_scale=guidance_scale,
                                           controlnet_conditioning_scale=conditioning_scale,
                                           num_inference_steps=num_inference_steps,
                                           generator=generator)

        title = break_up_string(prompt)

        fontsize = 12
        if len(title) / 50 > 3:
            fontsize = 10

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(source_image)
        ax[1].imshow(results.images[0])
        fig.suptitle(title, fontsize=fontsize, y=1)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        fig.tight_layout()

        if save_name:
            results.images[0].save(save_name)
        if comparison_save_name:
            fig.savefig(comparison_save_name)

        if display:
            plt.show()

        return results.images[0]

    # running the pipeline designed for interaction with the Unity Client that uses ARCore - AR Foundation
    def run_ARCore_pipeline(self, rgb_filepath, depth_filepath, i=0, prompt="",
                            guidance_scale=default_guidance_scale,
                            camIntrinsics="",
                            only_ground=False, display=False, save_eval=True):

        # if not prompt is given, randomly assign a prompt
        if not prompt:
            prompt_id = randrange(len(ModelData.PROMPT_LIST))
            prompt = ModelData.PROMPT_LIST[prompt_id]

        self.evaluator.set_prompt(prompt)

        # --- Setting all path variables and identifiers for storing results ---

        mesh_name = f"{i}_mesh.obj"
        material_name = f"{i}_mesh.obj.mtl"
        texture_name = f"{i}_mesh.png"
        relative_mesh_path_obj = os.path.join("processed_user_data", f"{i}_mesh.obj")
        full_mesh_path_obj = os.path.join(self.result_root, f"{relative_mesh_path_obj}")


        condition_id = self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        if self.multi_condition or not self.condition_type == "depth":
            condition_img_path = os.path.join(self.result_root, f"ControlNet/gen_depth_maps/{i}_gen_{condition_id}")

        identifier = f"{condition_id}_prompt_{prompt[0:min(5, len(prompt))]}_iter_{self.num_steps}_guide_{guidance_scale}_res_{self.resolution}"

        predict_depth_path = os.path.join(self.result_root, "ControlNet/gen_depth_maps",
                                          f"{i}_gen_depth_{identifier}")
        heatmap_path = os.path.join(self.result_root, f"ControlNet/depth_map_heatmaps",
                                    f"{i}_depth_heatmap_{identifier}.png")
        gen_pcd_path = os.path.join(self.result_root, f"ControlNet/gen_point_clouds",
                                    f"{i}_gen_pcd_{identifier}.ply")

        predict_ground_depth_path = os.path.join(self.result_root, "ControlNet", "predicted_ground_truth_depth_maps",
                                                 f"{i}_predict_ground_depth_res_{self.resolution}")

        ground_pcd_path = os.path.join(self.result_root, f"ground_point_clouds",
                                       f"{i}_ground_pcd_res_{self.resolution}.ply")

        gen_img_save_name = os.path.join(self.result_root, f"ControlNet", "2d_images",
                                         f"{i}_{identifier}.png")
        comparison_save_name = os.path.join(self.result_root, f"ControlNet", "2d_images",
                                            f"{i}_comparison_{identifier}.png")

        eval_table_path = os.path.join(self.result_root, f"ControlNet/eval_logs/{identifier}.csv")

        # ------------- Begin Processing
        start = time.time()

        rgb_image, arcore_depth_map, rgb_crop_val = prepare_arcore_data(rgb_filepath, depth_filepath)

        # setting the resolution to the minimum of the actual image resolution completed to a multiple of 64, or the
        # requested resolution
        self.resolution = min(rgb_image.shape[1] + 64 - rgb_image.shape[1] % 64, self.max_resolution)

        end = time.time()
        print(f"prepare_arcore_data | time: {end - start}")

        # getting parameters to use for all scaling and cropping stages
        H_original, W_original, H_resize, W_resize, scaling_factor, H_upsample, W_upsample = \
            get_sizing_params(resolution=self.resolution, original_shape=rgb_image.shape, round_down=False)

        # We make sure that all depth maps are the same shape as the final shape we will get after ControlNet inference
        # Controlnet input will be the first donwsampled to a multiple of 64 and then upsampled,
        # the final upsample dimensions must match

        rgb_image = reshape_to_shape(rgb_image, H_upsample, W_upsample, is_depth=False)
        arcore_depth_map = reshape_to_shape(arcore_depth_map, H_upsample, W_upsample, is_depth=True)

        # ---- Beginning Inference Steps

        # inferring the depth map of the original RGB image
        start = time.time()
        predict_ground_depth_map = self.infer_depth_map(rgb_image, save_name=predict_ground_depth_path, display=False)
        end = time.time()
        print(f"infer_depth_map | time: {end - start}")

        predict_ground_depth_map_aligned = align_midas_withzeros(predict_ground_depth_map, arcore_depth_map)

        # depth of the center point of the image is used for translating the center point of the mesh accordingly later
        c1, c2 = predict_ground_depth_map_aligned.shape
        center_depth = arcore_depth_map[c1 // 2, c2 // 2]

        print(f"Original center depth: {center_depth}")

        print(f"arcore_depth_map max: {arcore_depth_map.max()} | min: {arcore_depth_map.min()}")
        print(
            f"predict_ground_depth_map_aligned max: {predict_ground_depth_map_aligned.max()} | min: {predict_ground_depth_map_aligned.min()}")

        # constructing the point cloud for the original RGB image
        start = time.time()
        point_clouds.get_point_cloud(
            reshape_to_shape(rgb_image, H_original, W_original, is_depth=False),  # rgb_image,
            reshape_to_shape(predict_ground_depth_map_aligned, H_original, W_original, is_depth=True),
            # predict_ground_depth_map_aligned,
            camIntrinsics=camIntrinsics,
            pcd_path=ground_pcd_path,
            display=False)
        end = time.time()
        print(f"get_point_cloud | time: {end - start}")

        # ---- IF GENERATIVE MODE IS TURNED OFF ----
        if only_ground:
            center_depth = (
                                   predict_ground_depth_map_aligned.max() - predict_ground_depth_map_aligned.min()) / 2 - center_depth
            print(
                f"mean depth: {(predict_ground_depth_map_aligned.max() - predict_ground_depth_map_aligned.min()) / 2}")
            print(
                f"mean depth - center_depth: {center_depth}")
            return ground_pcd_path, center_depth

        # ---- IF GENERATIVE MODE IS ON ----

        start = time.time()

        # preparing condition inputs
        if self.multi_condition:
            depth_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map,
                                                                 scaling_factor=scaling_factor)
            seg_condition, _, _ = self.infer_seg_ade20k(rgb_image, i, scaling_factor=scaling_factor,
                                                        save_name=condition_img_path)
            ground_condition = [depth_condition, seg_condition]
        elif self.condition_type == "depth":
            ground_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map,
                                                                  scaling_factor=scaling_factor)
        else:  # segmentation
            ground_condition, H, W = self.infer_seg_ade20k(rgb_image, i, scaling_factor=scaling_factor,
                                                           save_name=condition_img_path)

        end = time.time()
        print(f"prepare depth | time: {end - start}")

        conditioning_scale = default_conditioning_scale if self.multi_condition else 1.0
        print(conditioning_scale)

        start = time.time()
        # Image generation
        gen_img = self.infer_controlnet(source_image=rgb_image, prompt=prompt, condition_image=ground_condition,
                                        H=H, W=W,
                                        guidance_scale=default_guidance_scale,
                                        conditioning_scale=conditioning_scale,
                                        num_inference_steps=self.num_steps,
                                        save_name=gen_img_save_name,
                                        comparison_save_name=comparison_save_name)

        end = time.time()
        print(f"infer_controlnet | time: {end - start}")

        gen_img_np = np.array(gen_img)
        # Upsampling the result image
        gen_img_np = resize_image(gen_img_np, resolution=W_upsample, is_depth=False, round=False, crop=False,
                                  ref_width=True)

        start = time.time()
        # Inferring the depth map of the generated image
        gen_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)
        gen_depth_map_aligned = align_midas_withzeros(gen_depth_map, arcore_depth_map)

        end = time.time()
        print(f"infer_depth_map gen | time: {end - start}")

        # creating a heatmap to visually compare depth maps
        vis.get_depth_heat_map(arcore_depth_map, predict_ground_depth_map_aligned, gen_depth_map_aligned, img_id=i,
                               save_name=heatmap_path)

        # constructing a point cloud from the generated RGB image
        start = time.time()
        point_clouds.get_point_cloud(reshape_to_shape(gen_img_np, H_original, W_original, is_depth=False),
                                     reshape_to_shape(gen_depth_map_aligned, H_original, W_original, is_depth=True),
                                     camIntrinsics=camIntrinsics,
                                     pcd_path=gen_pcd_path, display=display)
        end = time.time()
        print(f"get_point_cloud gen | time: {end - start}")

        # getting the difference between the median depth value of the center of the mesh bounding box
        # and the actual image center depth
        center_depth = (gen_depth_map_aligned.max() - gen_depth_map_aligned.min()) / 2 - center_depth

        # obtaining the Hausdorff distance result from point clouds
        mean_generation_dist = mesh_processing.get_hausdorff_distance(ground_pcd_path, gen_pcd_path)

        # creating the final mesh and obtaining the Hausdorff distance results between generated point cloud and mesh
        mean_meshing_dist = mesh_processing.process_mesh_marching_cubes(gen_pcd_path, full_mesh_path_obj, i,
                                                                        center_depth / 10)

        start = time.time()

        # preparing evaluation results
        eval_results = self.evaluator.evaluate_sample(rgb_image, gen_img_np, arcore_depth_map,
                                                      predict_ground_depth_map_aligned,
                                                      gen_depth_map_aligned,
                                                      mean_generation_dist, mean_meshing_dist, id=i,
                                                      save_path=eval_table_path if save_eval else "")
        end = time.time()
        print(f"evaluate_sample_aligned_noground | time: {end - start}")

        return mesh_name, material_name, texture_name

    # running the pipeline designed for benchmarking locally on the NYU dataset
    def run_NYU_pipeline(self, image, depth_map, i, prompt="",
                         guidance_scale=7.5, strength=0.5,
                         conditioning_scale=[1, 0.7],
                         display=False,
                         save_eval=True):
        src_img_np, ground_depth_map, _ = prepare_nyu_data(image, depth_map, image_resolution=None)

        prompt_id = randrange(len(ModelData.PROMPT_LIST))

        return self.run_pipeline(src_img_np, ground_depth_map, i, prompt=ModelData.PROMPT_LIST[prompt_id],
                                 guidance_scale=guidance_scale, strength=strength,
                                 conditioning_scale=conditioning_scale,
                                 dataset="NYU", display=display, save_eval=save_eval,
                                 prompt_id=prompt_id)  # , original_image_W=original_image_W)

    def run_pipeline(self, src_image, ground_depth_map, i, prompt="",
                     guidance_scale=7.5, strength=0.5,
                     conditioning_scale=[1.0, 0.7],
                     dataset="NYU",  # in case new datasets are added
                     display=False,
                     save_eval=True,
                     prompt_id=None,
                     original_image_W=None):

        # set new resolution for incoming image
        # because NYU images have W > H, to keep resolution effects comparable to the mobile application, we change resolution first:
        self.max_resolution = 640/480 * self.max_resolution

        # setting the resolution to the minimum of the actual image resolution completed to a multiple of 64, or the
        # requested resolution
        self.resolution = min(src_image.shape[1] + 64 - src_image.shape[1] % 64, self.max_resolution)

        # if not prompt is given, randomly assign a prompt
        if not prompt:
            prompt_id = randrange(len(ModelData.PROMPT_LIST))
            prompt = ModelData.PROMPT_LIST[prompt_id]

        self.evaluator.set_prompt(prompt)

        # --- Setting all path variables and identifiers for storing results ---

        condition_id = self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        controlnet_folder = os.path.join(self.result_root, "ControlNet")

        if self.multi_condition or not self.condition_type == "depth":
            condition_img_path = os.path.join(controlnet_folder, "ground_conditions", f"{i}_condition_{condition_id}_res_{self.resolution}")

        predict_ground_depth_path = os.path.join(controlnet_folder, f"predicted_ground_truth_depth_maps", f"{i}_predict_ground_depth_res_{self.resolution}")
        ground_pcd_path = os.path.join(self.result_root, f"ground_point_clouds", f"{i}_ground_pcd_res_{self.resolution}.ply")

        if prompt_id == None:
            prompt_id = prompt[0:min(5, len(prompt))]

        identifier = f"{condition_id}_prompt_{prompt_id}_iter_{self.num_steps}_guide_{guidance_scale}_res_{self.resolution}"
        # we want the prompt to vary within the evaluation process
        eval_identifier = f"{condition_id}_iter_{self.num_steps}_guide_{guidance_scale}_res_{self.resolution}"

        eval_table_path = os.path.join(controlnet_folder, "eval_logs", f"{eval_identifier}.csv")

        gen_img_save_name = os.path.join(controlnet_folder, "2d_images/", f"{i}_{identifier}.png")
        comparison_save_name = os.path.join(controlnet_folder, "2d_images", f"{i}_comparison_{identifier}.png")

        predict_depth_path = os.path.join(controlnet_folder, "gen_depth_maps", f"{i}_gen_depth_{identifier}")
        heatmap_path = os.path.join(controlnet_folder, "depth_map_heatmaps", f"{i}_depth_heatmap_{identifier}.png")
        gen_pcd_path = os.path.join(controlnet_folder, "gen_point_clouds", f"{i}_gen_pcd_{identifier}.ply")

        mesh_name = f"{i}_mesh.obj"
        material_name = f"{i}_mesh.obj.mtl"
        texture_name = f"{i}_mesh.png"
        relative_mesh_path_obj = os.path.join("meshes", mesh_name)
        full_mesh_path_obj = os.path.join(self.result_root, f"{relative_mesh_path_obj}")


        # getting parameters to use for all scaling and cropping stages
        H_original, W_original, H_resize, W_resize, scaling_factor, H_upsample, W_upsample = \
            get_sizing_params(resolution=self.resolution, original_shape=src_image.shape, round_down=False)

        # We make sure that all depth maps are the same shape as the final shape we will get after ControlNet inference
        # Controlnet input will be the first donwsampled to a multiple of 64 and then upsampled,
        # the final upsample dimensions must match

        src_image = reshape_to_shape(src_image, H_upsample, W_upsample, is_depth=False)
        ground_depth_map = reshape_to_shape(ground_depth_map, H_upsample, W_upsample, is_depth=True)

        # ---- Beginning Inference Steps

        # inferring the depth map of the original RGB image
        predict_ground_depth_map = self.infer_depth_map(src_image, save_name=predict_ground_depth_path,
                                                        display=display)
        predict_ground_depth_map_aligned = align_midas_withzeros(predict_ground_depth_map, ground_depth_map)

        # constructing the point cloud for the original RGB image
        original_pcd = point_clouds.get_point_cloud(
            reshape_to_shape(src_image, H_original, W_original, is_depth=False),  # rgb_image,
            reshape_to_shape(predict_ground_depth_map_aligned, H_original, W_original, is_depth=True),
            pcd_path=ground_pcd_path,
            display=display)

        # preparing condition inputs
        if self.multi_condition:
            depth_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map,
                                                                 scaling_factor=scaling_factor)
            seg_condition, _, _ = self.infer_seg_ade20k(src_image, i, scaling_factor=scaling_factor,
                                                        save_name=condition_img_path)
            ground_condition = [depth_condition, seg_condition]
        elif self.condition_type == "depth":
            ground_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map,
                                                                  scaling_factor=scaling_factor)
        else:  # segmentation
            ground_condition, H, W = self.infer_seg_ade20k(src_image, i, scaling_factor=scaling_factor,
                                                           save_name=condition_img_path)

        print(f"ground_depth_map max: {ground_depth_map.max()} | min: {ground_depth_map.min()}")
        print(
            f"predict_ground_depth_map_aligned max: {predict_ground_depth_map_aligned.max()} | min: {predict_ground_depth_map_aligned.min()}")

        # Image generation
        gen_img = self.infer_controlnet(source_image=src_image, prompt=prompt, condition_image=ground_condition,
                                        H=H, W=W,
                                        guidance_scale=guidance_scale,
                                        conditioning_scale=conditioning_scale if self.multi_condition else 1.0,
                                        num_inference_steps=self.num_steps,
                                        save_name=gen_img_save_name,
                                        comparison_save_name=comparison_save_name)

        gen_img_np = np.array(gen_img)
        # Upsampling the result image
        gen_img_np = resize_image(gen_img_np, resolution=W_upsample, is_depth=False, round=False, crop=False,
                                  ref_width=True)
        # Inferring the depth map of the generated image
        predict_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)
        predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        print(
            f"predict_depth_map_aligned max: {predict_depth_map_aligned.max()} | min: {predict_depth_map_aligned.min()}")

        # creating a heatmap to visually compare depth maps
        heatmap = vis.get_depth_heat_map(ground_depth_map, predict_ground_depth_map_aligned,
                                         predict_depth_map_aligned,
                                         img_id=i, save_name=heatmap_path)

        # constructing a point cloud from the generated RGB image
        generated_pcd = point_clouds.get_point_cloud(
            reshape_to_shape(gen_img_np, H_original, W_original, is_depth=False),
            reshape_to_shape(predict_depth_map_aligned, H_original, W_original, is_depth=True),
            pcd_path=gen_pcd_path, display=display)

        # obtaining the Hausdorff distance result from point clouds
        mean_generation_dist = mesh_processing.get_hausdorff_distance(ground_pcd_path, gen_pcd_path)

        # creating the final mesh and obtaining the Hausdorff distance results between generated point cloud and mesh
        mean_meshing_dist = mesh_processing.process_mesh_marching_cubes(gen_pcd_path, full_mesh_path_obj, i)

        # preparing evaluation results
        eval_results = self.evaluator.evaluate_sample(src_image, gen_img_np, ground_depth_map,
                                                      predict_ground_depth_map_aligned,
                                                      predict_depth_map_aligned,
                                                      mean_generation_dist, mean_meshing_dist,
                                                      id=i, prompt_id=prompt_id,
                                                      save_path=eval_table_path if save_eval else "")

        return eval_results


    def compute_macro_eval(self, prompt, guidance_scale, index_range=[0, 0]):

        condition_id = self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        identifier = f"{condition_id}_"
        if prompt:
            identifier += prompt[0:min(5, len(prompt))]
        identifier += f"iter_{self.num_steps}_guide_{guidance_scale}_res_{self.resolution}"

        macro_eval_path = self.result_root + f"ControlNet/eval_logs/macro_eval_metrics_{identifier}.csv"

        if index_range[1]:
            identifier = identifier + f"_range_{index_range[0]}-{index_range[1]}"
        else:
            identifier = identifier + f"_range_{index_range[0]}-end"

        self.evaluator.compute_macro_metrics(identifier=identifier, save_path=macro_eval_path)

# used for running benchmarking on the NYU dataset
def main(args):
    cache_dir = args.cache_dir

    data_path = args.data_path
    result_root = args.result_root
    f = h5py.File(data_path, "r")

    rgb_images = f['images']
    depth_maps = f['depths']

    pipeline = ControlNetModelWrapper(condition_type=args.condition_type,
                                      multi_condition=args.multi_condition,
                                      result_root=args.result_root,
                                      resolution=args.resolution,
                                      num_steps=args.num_steps,
                                      cache_dir=cache_dir)

    for i in range(0, rgb_images.shape[0]):

        try:
            pipeline.run_NYU_pipeline(rgb_images[i], depth_maps[i], i, prompt=args.prompt,
                                      guidance_scale=args.guidance_scale, strength=args.strength,
                                      display=args.display)

            if i > 0 and i % 5 == 0:
                pipeline.compute_macro_eval(prompt=args.prompt, guidance_scale=args.guidance_scale, index_range=[0, i])
        except Exception as e:
            print(f"------- For image no: {i} ------- ")
            print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/nyu_depth_v2_labeled.mat')
    parser.add_argument('--result_root', default='./results/NYU/')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--guidance_scale', default=default_guidance_scale, type=int)
    parser.add_argument('--strength', default=0.75, type=float)
    parser.add_argument('--num_steps', type=int, default=default_num_steps)
    parser.add_argument('--condition_type', type=str, default="depth")
    parser.add_argument('--multi_condition', type=bool, default=False)
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--resolution', type=int, default=512)

    args = parser.parse_args()
    main(args)
