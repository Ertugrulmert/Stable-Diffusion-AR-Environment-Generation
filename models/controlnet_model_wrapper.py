import argparse
import os
import sys

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
from models_3d import point_clouds

torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Defaults ------------
image_resolution = 512
depth_resolution = 512
num_steps = 20
guidance_scale = 9
seed = 1825989188


class ControlNetModelWrapper:

    def __init__(self, condition_type="seg", multi_condition=False,
                 depth_model_type="MiDaS",
                 result_root='./results/NYU/',
                 cache_dir=""):
        self.result_root = result_root
        self.multi_condition = multi_condition
        self.load_depth_model(model_type=depth_model_type, cache_dir=cache_dir)
        self.load_condition_model(condition_type=condition_type, cache_dir=cache_dir)
        self.load_controlnet(cache_dir=cache_dir)
        self.evaluator = Evaluator(condition_type=self.condition_type, cache_dir=cache_dir)

    # Model Loading

    def load_depth_model(self, model_type="MiDaS", cache_dir=""):
        # Loading Depth Model
        self.depth_model_type = model_type

        if cache_dir:
            torch.hub.set_dir(cache_dir + "/torch")

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

    def load_condition_model(self, condition_type="seg", cache_dir=""):

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        if self.multi_condition:
            self.condition_type = ["seg", "depth"]
        else:
            self.condition_type = condition_type

        if self.condition_type == "seg" or self.multi_condition:
            self.image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade",  **kwargs)
            self.segmentation_model = MaskFormerForInstanceSegmentation.from_pretrained(
                    "facebook/maskformer-swin-large-ade",  **kwargs)

    def load_controlnet(self, cache_dir=""):

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        torch.cuda.empty_cache()
        if self.multi_condition:
            self.controlnet_model_id = [ModelData.CONTROLNET_MODEL_IDS[type] for type in self.condition_type]
            controlnet = [ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)
                          for model_id in self.controlnet_model_id]
        else:
            self.controlnet_model_id = ModelData.CONTROLNET_MODEL_IDS[self.condition_type]
            controlnet = ControlNetModel.from_pretrained(self.controlnet_model_id, torch_dtype=torch.float32, **kwargs)

        self.coltrolnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            safety_checker=None,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            **kwargs)



        self.coltrolnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.coltrolnet_pipe.scheduler.config)
        # coltrolnet_pipe.enable_xformers_memory_efficient_attention()
        self.coltrolnet_pipe.to(device)
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
                # im = Image.fromarray(output).convert('RGB')
                # im.save(save_name + ".png")

            return output

    def infer_seg_ade20k(self, image, i, save_name=''):

        # pixel_values = image_processor(image, return_tensors="pt").pixel_values
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)

        temp_img = resize_image(image, image_resolution)
        H, W = temp_img.shape[:2]

        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

        for label, color in enumerate(ModelData.ade_palette):
            color_seg[seg == label, :] = color

        # temp_img = resize_image(color_seg, image_resolution)
        # H, W = temp_img.shape[:2]
        # color_seg = cv2.resize(color_seg, (W, H), interpolation=cv2.INTER_NEAREST)

        color_seg = color_seg.astype(np.uint8)
        control_image = Image.fromarray(color_seg)

        fontsize = 12

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(control_image)
        fig.suptitle(f"Predicted ADE20k Segmentation for Image {i}", fontsize=fontsize, y=0.9)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()

        if save_name:
            control_image.save(save_name + ".png")
        plt.show()

        return control_image, H, W

    def infer_controlnet(self, source_image, prompt, H, W, guidance_scale=7.5, conditioning_scale=1,
                         condition_image=None,
                         num_inference_steps=50, save_name='', comparison_save_name=''):

        generator = torch.Generator().manual_seed(seed)

        prompt = f'{prompt}, {ModelData.additional_prompt}'

        results = self.coltrolnet_pipe(prompt=prompt, negative_prompt=ModelData.negative_prompt,
                                       image=condition_image,
                                       # image=source_image,
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
                     conditioning_scale=[1, 0.7],
                     num_inference_steps=20,

                     save_eval=True):

        if not prompt:
            prompt = ModelData.interior_design_prompt_1

        self.evaluator.set_prompt(prompt)

        condition_id = self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        if self.multi_condition or not self.condition_type == "depth":
            condition_img_path = self.result_root + f"ControlNet/gen_depth_maps/{i}_gen_{condition_id}"

        predict_depth_path = self.result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth_from_{condition_id}"
        heatmap_path = self.result_root + f"ControlNet/depth_map_heatmaps/{i}_depth_heatmap_from_{condition_id}.png"
        gen_pcd_path = self.result_root + f"ControlNet/gen_point_clouds/{i}_gen_pcd_from_{condition_id}"

        # else:
        #    predict_depth_path = self.result_root + f"depth_maps/{i}_gen_depth"
        #    heatmap_path = self.result_root + f"depth_map_heat_maps/{i}_depth_heatmap.png"
        #    gen_pcd_path = self.result_root + f"point_clouds/{i}_gen_pcd"

        predict_ground_depth_path = self.result_root + f"ControlNet/predicted_ground_truth_depth_maps/{i}_predict_ground_depth"

        ground_pcd_path = self.result_root + f"ground_point_clouds/{i}_ground_pcd"
        view_setting_path = self.result_root + "view_setting.json"

        gen_img_save_name = self.result_root + f"ControlNet/2d_images/{i}_generated_from_{condition_id}.png"
        comparison_save_name = self.result_root + f"ControlNet/2d_images/{i}_comparison_from_{condition_id}.png"

        identifier = f"{condition_id}_prompt_{prompt[0:min(5, len(prompt))]}_iter_{num_inference_steps}_guide_{guidance_scale}.csv"
        eval_table_path = self.result_root + f"ControlNet/eval_logs/{identifier}.csv"

        # src_img_np, ground_condition_np = prepare_nyu_data(image, condition_data[i])
        src_img_np, ground_depth_map = prepare_nyu_data(image, depth_map, image_resolution=image_resolution)

        # if self.condition_type == "depth":
        #    ground_depth_map = ground_condition_np
        # else:
        #   _, ground_depth_map = prepare_nyu_data(condition_img=depth_map)

        # ground_for_heatmap = resize_image(ground_depth_map, image_resolution)

        predict_ground_depth_map = self.infer_depth_map(src_img_np, save_name=predict_ground_depth_path)
        predict_ground_depth_map = resize_image(predict_ground_depth_map, image_resolution)
        predict_ground_depth_map_aligned = align_midas(predict_ground_depth_map, ground_depth_map)

        if self.multi_condition:
            depth_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map)
            seg_condition, _, _ = self.infer_seg_ade20k(src_img_np, i, save_name=condition_img_path)
            ground_condition = [depth_condition, seg_condition]
        elif self.condition_type == "depth":
            ground_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map)
        else:  # seg
            # ground_condition, H, W = prepare_nyu_controlnet_seg(ground_condition_np, num_classes=num_classes)
            ground_condition, H, W = self.infer_seg_ade20k(src_img_np, i, save_name=condition_img_path)

        gen_img = self.infer_controlnet(source_image=src_img_np, prompt=prompt, condition_image=ground_condition,
                                        H=H, W=W,
                                        guidance_scale=guidance_scale,
                                        conditioning_scale=conditioning_scale if self.multi_condition else 1,
                                        num_inference_steps=num_inference_steps,
                                        save_name=gen_img_save_name,
                                        comparison_save_name=comparison_save_name)

        gen_img_np = np.array(gen_img)

        # ground_depth_map = infer_depth_map(source_img_np)

        predict_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)
        predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        heatmap = vis.get_depth_heat_map(ground_depth_map, predict_ground_depth_map_aligned,
                                         predict_depth_map_aligned,
                                         img_id=i, save_name=heatmap_path)

        original_pcd = point_clouds.get_point_cloud(src_img_np, ground_depth_map, pcd_path=ground_pcd_path + ".pcd")
        # original_pcd = get_point_cloud(src_img_np, ground_depth_map,  pcd_path=ground_pcd_path+".pcd")
        generated_pcd = point_clouds.get_point_cloud(gen_img_np, predict_depth_map_aligned,
                                                     pcd_path=gen_pcd_path + ".pcd")

        vis.capture_pcd_with_view_params(pcd=original_pcd, pcd_path=ground_pcd_path + ".png",
                                         view_setting_path=view_setting_path)
        vis.capture_pcd_with_view_params(pcd=generated_pcd, pcd_path=gen_pcd_path + ".png",
                                         view_setting_path=view_setting_path)

        eval_results = self.evaluator.evaluate_sample(src_img_np, gen_img_np, ground_depth_map,
                                                      predict_ground_depth_map_aligned,
                                                      predict_depth_map_aligned, id=i,
                                                      save_path=eval_table_path if save_eval else "")

        # if save_eval:
        #    eval_results.to_csv(eval_table_path, mode='a', index=False, header=False)

        return eval_results

    def run_noground_pipeline(self, image, i, prompt="",
                              guidance_scale=7.5, strength=0.5,
                              conditioning_scale=[1, 0.7],
                              num_inference_steps=20,
                              save_eval=True):

        if not prompt:
            prompt = ModelData.interior_design_prompt_1

        self.evaluator.set_prompt(prompt)

        condition_id = "_ng_" + self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        if self.multi_condition or not self.condition_type == "depth":
            condition_img_path = self.result_root + f"depth_maps/{i}_gen_{condition_id}"

        predict_depth_path = self.result_root + f"depth_maps/{i}_gen_depth_from_{condition_id}"
        heatmap_path = self.result_root + f"depth_map_heatmaps/{i}_depth_heatmap_from_{condition_id}.png"
        gen_pcd_path = self.result_root + f"point_clouds/{i}_gen_pcd_from_{condition_id}"

        # else:
        #    predict_depth_path = self.result_root + f"depth_maps/{i}_gen_depth"
        #    heatmap_path = self.result_root + f"depth_map_heat_maps/{i}_depth_heatmap.png"
        #    gen_pcd_path = self.result_root + f"point_clouds/{i}_gen_pcd"

        predict_ground_depth_path = self.result_root + f"predicted_ground_truth_depth_maps/{i}_predict_ground_depth"

        ground_pcd_path = self.result_root + f"point_clouds/{i}_ground_pcd"
        view_setting_path = self.result_root + "view_setting.json"

        gen_img_save_name = self.result_root + f"2d_images/{i}_generated_from_{condition_id}.png"
        comparison_save_name = self.result_root + f"2d_images/{i}_comparison_from_{condition_id}.png"

        identifier = f"{condition_id}_prompt_{prompt[0:min(5, len(prompt))]}_iter_{num_inference_steps}_guide_{guidance_scale}"
        eval_table_path = self.result_root + f"eval_logs/{identifier}.csv"

        # src_img_np, ground_condition_np = prepare_nyu_data(image, condition_data[i])
        src_img_np, _ = prepare_nyu_data(rgb_img=image, image_resolution=image_resolution)

        predict_ground_depth_map = self.infer_depth_map(src_img_np, save_name=predict_ground_depth_path)
        predict_ground_depth_map = resize_image(predict_ground_depth_map, image_resolution)

        if self.multi_condition:
            depth_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map)
            seg_condition, _, _ = self.infer_seg_ade20k(src_img_np, i, save_name=condition_img_path)
            ground_condition = [depth_condition, seg_condition]
        elif self.condition_type == "depth":
            ground_condition, H, W = prepare_nyu_controlnet_depth(predict_ground_depth_map)
        else:  # seg
            # ground_condition, H, W = prepare_nyu_controlnet_seg(ground_condition_np, num_classes=num_classes)
            ground_condition, H, W = self.infer_seg_ade20k(src_img_np, i, save_name=condition_img_path)

        gen_img = self.infer_controlnet(source_image=src_img_np, prompt=prompt, condition_image=ground_condition,
                                        H=H, W=W,
                                        guidance_scale=guidance_scale,
                                        conditioning_scale=conditioning_scale if self.multi_condition else 1,
                                        num_inference_steps=num_inference_steps,
                                        save_name=gen_img_save_name,
                                        comparison_save_name=comparison_save_name)

        gen_img_np = np.array(gen_img)

        # ground_depth_map = infer_depth_map(source_img_np)

        predict_depth_map = self.infer_depth_map(gen_img_np, save_name=predict_depth_path)
        # predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        # heatmap = vis.get_depth_heat_map(ground_depth_map, predict_ground_depth_map_aligned,
        # predict_depth_map_aligned,
        # img_id=i, save_name=heatmap_path)

        inverted_predict_ground_depth = 1 / (predict_ground_depth_map + 10e-6)
        inverted_predict_depth = 1 / (predict_depth_map + 10e-6)

        original_pcd = point_clouds.get_point_cloud(src_img_np, inverted_predict_ground_depth,
                                                    pcd_path=ground_pcd_path + ".pcd")
        # original_pcd = get_point_cloud(src_img_np, ground_depth_map,  pcd_path=ground_pcd_path+".pcd")
        generated_pcd = point_clouds.get_point_cloud(gen_img_np, inverted_predict_depth,
                                                     pcd_path=gen_pcd_path + ".pcd")

        vis.capture_pcd_with_view_params(pcd=original_pcd, pcd_path=ground_pcd_path + ".png",
                                         view_setting_path=view_setting_path)
        vis.capture_pcd_with_view_params(pcd=generated_pcd, pcd_path=gen_pcd_path + ".png",
                                         view_setting_path=view_setting_path)

        eval_results = self.evaluator.evaluate_sample(src_img_np, gen_img_np, inverted_predict_ground_depth,
                                                      inverted_predict_ground_depth,
                                                      inverted_predict_depth, id=i,
                                                      save_path=eval_table_path if save_eval else "")

        # if save_eval:
        #    eval_results.to_csv(eval_table_path, mode='a', index=False, header=False)

        return eval_results

    def compute_macro_eval(self, prompt, num_inference_steps, guidance_scale, index_range=[0, 0]):

        condition_id = self.condition_type if not self.multi_condition else f"_{'_'.join(self.condition_type)}_"

        identifier = f"{condition_id}_prompt_{prompt[0:min(5, len(prompt))]}_iter_{num_inference_steps}_guide_{guidance_scale}"

        if index_range[1]:
            identifier = identifier + f"_range_{index_range[0]}-{index_range[1]}"
        else:
            identifier = identifier + f"_range_{index_range[0]}-end"

        macro_eval_path = self.result_root + "eval_logs/macro_eval_metrics.csv"

        self.evaluator.compute_macro_metrics(identifier=identifier, save_path=macro_eval_path)


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
                                      cache_dir=cache_dir)

    for i in range(0, rgb_images.shape[0]):
        pipeline.run_pipeline(rgb_images[i], depth_maps[i], i, prompt=args.prompt,
                              guidance_scale=args.guidance_scale, strength=args.strength,
                              num_inference_steps=args.num_inference_steps)

        if i > 0 and i % 5 == 0:
            pipeline.compute_macro_eval(prompt=args.prompt, num_inference_steps=args.num_inference_steps,
                                        guidance_scale=args.guidance_scale, index_range=[0, i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/nyu_depth_v2_labeled.mat')
    parser.add_argument('--result_root', default='./results/NYU/')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--guidance_scale', default=9, type=int)
    parser.add_argument('--strength', default=0.75, type=float)
    parser.add_argument('--num_inference_steps', type=int, default=40)
    parser.add_argument('--condition_type', type=str, default="depth")
    parser.add_argument('--multi_condition', type=bool, default=False)
    parser.add_argument('--cache_dir', type=str, default="")

    args = parser.parse_args()
    main(args)
