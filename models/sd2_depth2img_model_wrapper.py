from PIL import Image
import numpy as np
from io import BytesIO
import os
from diffusers import  StableDiffusionDepth2ImgPipeline

from utils.evaluation import Evaluator
from utils.preprocessing import *
from models.model_data import *
from utils.visualisation import Visualiser as vis
from models_3d import point_clouds

torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL_ID = "stabilityai/stable-diffusion-2-depth"

# Defaults ------------
image_resolution = 512
depth_resolution = 512
num_steps = 30
guidance_scale = 9
seed = 1825989188

class Depth2ImgWrapper:

    def __init__(self, depth_model_type="MiDaS",
                 result_root='./results/NYU/'):
        self.result_root = result_root
        self.load_depth_model(model_type=depth_model_type)
        self.load_model()
        self.evaluator = Evaluator(condition_type="depth")

    # Model Loading

    def load_depth_model(self, model_type="MiDaS"):
        # Loading Depth Model
        self.depth_model_type = model_type

        if self.depth_model_type == "MiDaS":
            self.depth_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
            self.depth_model.to(device)
            self.depth_model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.depth_model_type == "DPT_Large" or self.depth_model_type == "DPT_Hybrid":
            self.depth_transform = midas_transforms.dpt_transform
        elif self.depth_model_type == "MiDaS":
            self.depth_transform = midas_transforms.default_transform
        else:
            self.depth_transform = midas_transforms.small_transform

    def load_model(self):

        self.model_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            MODEL_ID
            #torch_dtype=torch.float16,
        ).to(device)
        torch.cuda.empty_cache()

    def infer_depth_map(self, image, save_name='', display=True):

        input_batch = self.depth_transform(image).to(device)

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
                #im = Image.fromarray(output).convert('RGB')
                #im.save(save_name + ".png")

            return output


    def infer_model(self, source_image, prompt,guidance_scale=9.0,
                         num_inference_steps=50, save_name='', comparison_save_name=''):

        generator = torch.Generator().manual_seed(seed)

        prompt = f'{prompt}, {ModelData.additional_prompt}'

        results = self.model_pipe(prompt=prompt, negative_prompt=ModelData.negative_prompt,
                                          image=source_image,
                                          guidance_scale=guidance_scale,
                                          num_inference_steps=num_inference_steps,
                                          generator=generator)

        title = break_up_string(prompt)

        fontsize = 12
        if len(title) / 50 > 3:
            fontsize = 10

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(source_image)
        ax[1].imshow(results.images[0])
        fig.suptitle(title, fontsize=fontsize, y=0.9)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        fig.tight_layout()

        if save_name:
            results.images[0].save(save_name)
        if comparison_save_name:
            fig.savefig(comparison_save_name)

        plt.show()

        return results.images[0]

    def run_pipeline(self, image, depth_map, i, prompt="",
                     guidance_scale=7.5, strength=0.5,
                     num_inference_steps=20,

                     save_eval=True):

        if not prompt:
            prompt = ModelData.interior_design_prompt_1

        self.evaluator.set_prompt(prompt)

        condition_id = "sd2_depth2img"

        predict_depth_path = self.result_root + f"SD2_depth2img/gen_depth_maps/{i}_gen_depth_from_{condition_id}"
        heatmap_path = self.result_root + f"SD2_depth2img/depth_map_heatmaps/{i}_depth_heatmap_from_{condition_id}.png"
        gen_pcd_path = self.result_root + f"SD2_depth2img/gen_point_clouds/{i}_gen_pcd_from_{condition_id}"

        predict_ground_depth_path = self.result_root + f"SD2_depth2img/predicted_ground_truth_depth_maps/{i}_predict_ground_depth"

        ground_pcd_path = self.result_root + f"ground_point_clouds/{i}_ground_pcd"
        view_setting_path = self.result_root + "view_setting.json"

        gen_img_save_name = self.result_root + f"SD2_depth2img/2d_images/{i}_generated_from_{condition_id}.png"
        comparison_save_name = self.result_root + f"SD2_depth2img/2d_images/{i}_comparison_from_{condition_id}.png"

        identifier = f"{condition_id}_prompt_{prompt[0:min(5,len(prompt))]}_iter_{num_inference_steps}_guide_{guidance_scale}.csv"
        eval_table_path = self.result_root + f"SD2_depth2img/eval_logs/{identifier}.csv"

        src_img_np, ground_depth_map = prepare_nyu_data(image, depth_map, image_resolution=image_resolution)

        predict_ground_depth_map = self.infer_depth_map(src_img_np, save_name=predict_ground_depth_path)
        predict_ground_depth_map = resize_image(predict_ground_depth_map, image_resolution)
        predict_ground_depth_map_aligned = align_midas(predict_ground_depth_map, ground_depth_map)

        src_img = Image.fromarray(np.uint8(src_img_np))

        gen_img = self.infer_model(source_image=src_img, prompt=prompt,
                                        guidance_scale=guidance_scale,
                                        num_inference_steps=num_inference_steps,
                                        save_name=gen_img_save_name,
                                        comparison_save_name=comparison_save_name)

        gen_img_np = np.array(gen_img)

        predict_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)
        predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        heatmap = vis.get_depth_heat_map(ground_depth_map, predict_ground_depth_map_aligned,
                                         predict_depth_map_aligned,
                                         img_id=i, save_name=heatmap_path)

        original_pcd = point_clouds.get_point_cloud(src_img_np, ground_depth_map, pcd_path=ground_pcd_path + ".pcd")
        generated_pcd = point_clouds.get_point_cloud(gen_img_np, predict_depth_map_aligned,
                                                     pcd_path=gen_pcd_path + ".pcd")

        vis.capture_pcd_with_view_params(pcd=original_pcd, pcd_path=ground_pcd_path + ".png",
                                         view_setting_path=view_setting_path)
        vis.capture_pcd_with_view_params(pcd=generated_pcd, pcd_path=gen_pcd_path + ".png",
                                         view_setting_path=view_setting_path)

        eval_results = self.evaluator.evaluate_sample(src_img_np, gen_img_np, ground_depth_map, predict_ground_depth_map_aligned,
                    predict_depth_map_aligned, id=i)

        if save_eval:
            eval_results.to_csv(eval_table_path, mode='a', index=False, header=False)

        return eval_results
