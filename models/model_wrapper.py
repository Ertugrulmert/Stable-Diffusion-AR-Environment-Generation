import torch
# from torch import autocast // only for GPU

from PIL import Image
import numpy as np
from io import BytesIO
import os
from diffusers import StableDiffusionImg2ImgPipeline
from utils.preprocessing import *

TOKEN = os.environ.get('HF_TOKEN_SD')
device = "cuda"

project_path = "../"


class ModelWrapper:
    data_path = "../" + "data"
    result_path = "../" + "results"
    strength = 0.75  # 0-1
    guidance_scale = 10  # 2-15 rec: 7
    num_inference_steps = 10  # 10-50

    def __init__(self, data_path="../" + "data",
                 result_path="../" + "results"):
        self.data_path = data_path
        self.result_path = result_path
        self.model = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                    use_auth_token=TOKEN)
        self.model.to(device)

    def infer(self, image, prompt):
        input_image = prepare_img(image)

        images_list = self.model([prompt] * 1, init_image=image, strength=self.strength, guidance_scale=self.guide,
                                 num_inference_steps=self.steps)

        return images_list
