import pandas as pd
import csv
import torch
import h5py
from PIL import Image

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.preprocessing import prepare_nyu_data, align_midas
from models.model_data import *


# taken from https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class Evaluator:
    columns = ['id', 'LPIPS', 'CLIPScore',
               "GR-Abs-Rel", "GR-Sqr-Rel", "GR-RMSE", "GR-RMSE-log", "GR-thresh-1", "GR-thresh-2", "GR-thresh-3", \
               "Pred-GR-Abs-Rel", "Pred-GR-Sqr-Rel", "Pred-GR-RMSE", "Pred-GR-RMSE-log", \
               "Pred-GR-thresh-1", "Pred-GR-thresh-2", "Pred-GR-thresh-3", \
               "Pred-Gen-Abs-Rel", "Pred-Gen-Sqr-Rel", "Pred-Gen-RMSE", "Pred-Gen-RMSE-log", \
               "Pred-Gen-thresh-1", "Pred-Gen-thresh-2", "Pred-Gen-thresh-3"]

    def __init__(self, condition_type="seg", prompt=ModelData.interior_design_prompt_1, cache_dir=""):

        if cache_dir:
            torch.hub.set_dir(cache_dir + "/torch")

        self.condition_type = condition_type
        self.prompt = self.set_prompt(prompt)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.fid = FrechetInceptionDistance(feature=64)
        self.inception = InceptionScore()
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")

    def set_prompt(self, prompt, token_limit=70):
        tokens = prompt.split()
        tokens = tokens[0:min(len(tokens), token_limit)]
        self.prompt = ' '.join(tokens)

    def evaluate_set(self, dataset_path, result_root, condition_type="seg", prompt=ModelData.interior_design_prompt_1,
                     index_range=(0, 0)):

        self.set_prompt(prompt)
        df = pd.DataFrame(columns=self.columns)

        # lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        # fid = FrechetInceptionDistance(feature=64)
        # inception = InceptionScore()
        # clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")

        f = h5py.File(dataset_path)
        rgb_images = f['images']
        depth_maps = f['depths']

        index_begin = index_range[0] if index_range[0] > 0 else 0
        index_end = index_range[1] if 0 < index_range[0] <= rgb_images.shape[0] else rgb_images.shape[0]

        # index_limit = min(index_limit,rgb_images.shape[0])

        src_images = []
        gen_images = []
        ground_depth_images = []
        predict_ground_depth_images = []
        gen_depth_images = []

        identifier = f"{condition_type}_range_{index_begin}-{index_end}"

        eval_table_path = result_root + f"eval_logs/{identifier}.csv"

        macro_eval_path = result_root + "eval_logs/macro_eval_metrics.csv"

        for i in range(index_begin, index_end):

            if not condition_type == "depth":
                condition_img_path = result_root + f"depth_maps/{i}_gen_{condition_type}"
                predict_depth_path = result_root + f"depth_maps/{i}_gen_depth_from_{condition_type}.npy"
                heatmap_path = result_root + f"depth_map_heat_maps/{i}_depth_heatmap_from_{condition_type}.png"
                gen_pcd_path = result_root + f"point_clouds/{i}_gen_pcd_from_{condition_type}"
            else:
                predict_depth_path = result_root + f"depth_maps/{i}_gen_depth.npy"
                heatmap_path = result_root + f"depth_map_heat_maps/{i}_depth_heatmap.png"
                gen_pcd_path = result_root + f"point_clouds/{i}_gen_pcd"

            predict_ground_depth_path = result_root + f"predicted_ground_truth_depth_maps/{i}_predict_ground_depth.npy"

            ground_pcd_path = result_root + f"point_clouds/{i}_ground_pcd"
            view_setting_path = result_root + "view_setting.json"

            gen_img_path = result_root + f"2d_images/{i}_generated_from_{condition_type}.png"

            predict_ground_depth_map = np.load(predict_ground_depth_path)
            predict_depth_map = np.load(predict_depth_path)
            image_resolution = predict_depth_map.shape[0]

            src_img_np, ground_depth_map = prepare_nyu_data(rgb_images[i], depth_maps[i],
                                                            image_resolution=image_resolution)

            gen_img = Image.open(gen_img_path)
            gen_img_np = np.array(gen_img)

            predict_ground_depth_map_aligned = align_midas(predict_ground_depth_map, ground_depth_map)
            predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

            # creating image tensors
            src_img_tensor = torch.from_numpy(np.moveaxis(src_img_np, 2, 0)).unsqueeze(0)
            gen_img_tensor = torch.from_numpy(np.moveaxis(gen_img_np, 2, 0)).unsqueeze(0)

            ground_depth_tensor = torch.from_numpy(ground_depth_map).unsqueeze(0)
            predict_ground_depth_tensor = torch.from_numpy(predict_ground_depth_map_aligned).unsqueeze(0)
            gen_depth_tensor = torch.from_numpy(predict_depth_map_aligned).unsqueeze(0)

            # -- RGB Image Generation Evaluation

            lpips_score = self.lpips(src_img_tensor.float() / 255, gen_img_tensor.float() / 255)
            clip_score = self.clip(gen_img_tensor, self.prompt).item()

            # -- Depth Map Evaluation

            # -- Ground vs Predicted Ground

            g_pg_abs_rel, g_pg_sq_rel, g_pg_rmse, g_pg_rmse_log, g_pg_a1, g_pg_a2, g_pg_a3 = compute_errors(
                ground_depth_map, predict_ground_depth_map_aligned)

            # -- Predicted Ground vs Predicted Generated

            pg_pgen_abs_rel, pg_pgen_sq_rel, pg_pgen_rmse, pg_pgen_rmse_log, pg_pgen_a1, pg_pgen_a2, pg_pgen_a3 = compute_errors(
                predict_ground_depth_map_aligned, predict_depth_map_aligned)

            # -- Ground vs Predicted Generated

            g_pgen_abs_rel, g_pgen_sq_rel, g_pgen_rmse, g_pgen_rmse_log, g_pgen_a1, g_pgen_a2, g_pgen_a3 = compute_errors(
                ground_depth_map, predict_depth_map_aligned)

            df.loc[i] = [i,  # fid_score,
                         lpips_score.item(),
                         # inception_score,
                         clip_score,
                         g_pg_abs_rel, g_pg_sq_rel, g_pg_rmse, g_pg_rmse_log, g_pg_a1, g_pg_a2, g_pg_a3,
                         pg_pgen_abs_rel, pg_pgen_sq_rel, pg_pgen_rmse, pg_pgen_rmse_log, pg_pgen_a1, pg_pgen_a2,
                         pg_pgen_a3,
                         g_pgen_abs_rel, g_pgen_sq_rel, g_pgen_rmse, g_pgen_rmse_log, g_pgen_a1, g_pgen_a2, g_pgen_a3]

            # storing tensors for metric that require multiple samples
            src_images.append(src_img_tensor)
            gen_images.append(gen_img_tensor)

            # ground_depth_images.append(ground_depth_tensor)
            # predict_ground_depth_images.append(predict_ground_depth_tensor)
            # gen_depth_images.append(gen_depth_tensor)

        src_image_tensor = torch.cat(src_images)
        gen_image_tensor = torch.cat(gen_images)

        self.fid.update(src_image_tensor, real=True)
        self.fid.update(gen_image_tensor, real=False)
        fid_score = self.fid.compute()

        self.inception.update(gen_image_tensor)
        inception_score = self.inception.compute()

        df.to_csv(eval_table_path, index=False)

        with open(macro_eval_path, 'a') as csvfile:
            fieldnames = ['identifier', 'FID', 'IS_mean', 'IS_std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'identifier': identifier, 'FID': fid_score.item(),
                             'IS_mean': inception_score[0].item(),
                             'IS_std': inception_score[1].item()})

        return df, fid_score, inception_score

    def evaluate_sample(self, src_img_np, gen_img_np, ground_depth_map, predict_ground_depth_map, predict_depth_map,
                        prompt="", id=0, save_path=""):
        if prompt:
            self.set_prompt(prompt)
        df = pd.DataFrame(columns=self.columns)

        predict_ground_depth_map_aligned = align_midas(predict_ground_depth_map, ground_depth_map)
        predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

        # creating image tensors
        src_img_tensor = torch.from_numpy(np.moveaxis(src_img_np, 2, 0)).unsqueeze(0)
        gen_img_tensor = torch.from_numpy(np.moveaxis(gen_img_np, 2, 0)).unsqueeze(0)

        # these metrics require full dataset so they will be computed after all iterations are over
        self.fid.update(src_img_tensor, real=True)
        self.fid.update(gen_img_tensor, real=False)
        self.inception.update(gen_img_tensor)

        # -- RGB Image Generation Evaluation

        lpips_score = self.lpips(src_img_tensor.float() / 255, gen_img_tensor.float() / 255)
        clip_score = self.clip(gen_img_tensor, self.prompt).item()

        # -- Depth Map Evaluation

        # -- Ground vs Predicted Ground

        g_pg_abs_rel, g_pg_sq_rel, g_pg_rmse, g_pg_rmse_log, g_pg_a1, g_pg_a2, g_pg_a3 = compute_errors(
            ground_depth_map, predict_ground_depth_map_aligned)

        # -- Predicted Ground vs Predicted Generated

        pg_pgen_abs_rel, pg_pgen_sq_rel, pg_pgen_rmse, pg_pgen_rmse_log, pg_pgen_a1, pg_pgen_a2, pg_pgen_a3 = compute_errors(
            predict_ground_depth_map_aligned, predict_depth_map_aligned)

        # -- Ground vs Predicted Generated

        g_pgen_abs_rel, g_pgen_sq_rel, g_pgen_rmse, g_pgen_rmse_log, g_pgen_a1, g_pgen_a2, g_pgen_a3 = compute_errors(
            ground_depth_map, predict_depth_map_aligned)

        df.loc[0] = [id,
                     lpips_score.item(),
                     clip_score,
                     g_pg_abs_rel, g_pg_sq_rel, g_pg_rmse, g_pg_rmse_log, g_pg_a1, g_pg_a2, g_pg_a3,
                     pg_pgen_abs_rel, pg_pgen_sq_rel, pg_pgen_rmse, pg_pgen_rmse_log, pg_pgen_a1, pg_pgen_a2,
                     pg_pgen_a3,
                     g_pgen_abs_rel, g_pgen_sq_rel, g_pgen_rmse, g_pgen_rmse_log, g_pgen_a1, g_pgen_a2, g_pgen_a3]

        if save_path:
            df.to_csv(save_path, mode='a', index=False, header=False)

        return df

    def compute_macro_metrics(self, identifier="full", save_path=""):
        fid_score = self.fid.compute()
        inception_score = self.inception.compute()

        columns = ['identifier', 'FID', 'IS_mean', 'IS_std']
        df = pd.DataFrame(columns=columns)
        df.loc[0] = [identifier, fid_score.item(), inception_score[0].item(), inception_score[1].item()]

        if save_path:
            df.to_csv(save_path, mode='a', index=False, header=False)

        return df
