import os
import urllib.request
from io import BytesIO
import requests
import json

import torch

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import h5py

import open3d as o3d

import matplotlib.image as mpimg
import re
from csv import writer


def extractDepth(x):
    depthConfidence = (x >> 13) & 0x7
    if (depthConfidence > 6): return 0
    return x & 0x1FFF


def crop_image(image, crop_rate, croph=True, cropw=True):
    if crop_rate <= 0 or crop_rate >= 0.5:
        return image

    h = image.shape[0]
    w = image.shape[1]
    crop_h = int(h * crop_rate) if croph else 0
    crop_w = int(w * crop_rate) if cropw else 0
    return image[crop_h:(h - crop_h), crop_w:(w - crop_w)]


def prepare_arcore_data(rgb_filepath, depth_filepath, confidence_filepath=None, image_resolution=512, crop_rate=0,
                        depth_H=90, depth_W=160, crop=False):
    rgb_image = cv2.imread(rgb_filepath)
    rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
    rgb_image = cv2.flip(rgb_image, 0)
    rgb_image = cv2.flip(rgb_image, 1)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    depthData = np.fromfile(depth_filepath, dtype=np.uint16)
    depthMap = np.array([extractDepth(x) for x in depthData]).reshape(depth_H, depth_W)
    depthMap = cv2.rotate(depthMap, cv2.ROTATE_90_CLOCKWISE)

    print(f"prepare_arcore_data depthMap max: {depthMap.max()} | min: {depthMap.min()}")

    depthMap = np.float32(depthMap)
    depthMap /= 1000

    rgb_H = rgb_image.shape[0]
    # depth image must be resized initially to have the same height as color image
    print(f"before initial resize depth: {depthMap.shape}")
    depth_resized = resize_image(depthMap, rgb_H, ref_width=False, is_depth=True, round=False)
    print(f"after initial resize depth: {depth_resized.shape}")
    d_w = depth_resized.shape[1]
    rgb_w = rgb_image.shape[1]

    print(f"before initial crop rgb: {rgb_image.shape}")

    if crop:
        rgb_crop_val = int((rgb_w - d_w) / 2)
        rgb_image = rgb_image[:, rgb_crop_val:rgb_w - rgb_crop_val, :]
        print(f"initial cropped rgb: {rgb_image.shape}")

    else:
        rgb_crop_val=0
        pad_val = int((rgb_w - d_w) / 2)
        depth_resized = np.pad(depth_resized, ((0, 0), (pad_val, pad_val)), mode='constant')

    return rgb_image, depth_resized, rgb_crop_val


def prepare_nyu_data(rgb_img=None, condition_img=None, image_resolution=512):
    img_ = None
    original_image_W = image_resolution
    if rgb_img is not None:
        # reshape
        img_ = np.empty([rgb_img.shape[2], rgb_img.shape[1], 3])
        img_[:, :, 0] = rgb_img[0, :, :].T
        img_[:, :, 1] = rgb_img[1, :, :].T
        img_[:, :, 2] = rgb_img[2, :, :].T
        img_ = img_[10:-9, 10:-9]
        original_image_W = img_.shape[1]
        if image_resolution:
            original_image_W = img_.shape[1] - img_.shape[1] % 64
            img_ = resize_image(img_, image_resolution, crop=True)

        img_ = img_.astype(np.uint8)

    condition_np = None
    if condition_img is not None:
        condition_np = np.asarray(condition_img.T, dtype=np.float32, order="C")
        condition_np = condition_np.astype(np.float32)
        condition_np = condition_np[10:-9, 10:-9]
        if image_resolution:
            condition_np = resize_image(condition_np, image_resolution, is_depth=True, crop=True)

    return img_, condition_np, original_image_W


def align_midas(midas_pred, ground_truth):

    ground_truth_invert = 1 / (ground_truth + 10e-6)  # invert absolute depth with meters
    x = midas_pred.copy().flatten()  # Midas Depth
    y = ground_truth_invert.copy().flatten()  # Realsense invert Depth
    A = np.vstack([x, np.ones(len(x))]).T
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    midas_aligned_invert = midas_pred * s + t
    midas_aligned = 1 / (midas_aligned_invert + 10e-6)

    return midas_aligned


def align_midas_withzeros(midas_pred, ground_truth):
    nonzero = np.nonzero(ground_truth)

    ground_truth_invert = 1 / (ground_truth[nonzero] + 10e-6)  # invert absolute depth with meters
    x = midas_pred.copy()[nonzero].flatten()  # Midas Depth
    y = ground_truth_invert.copy().flatten()
    A = np.vstack([x, np.ones(len(x))]).T
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    midas_aligned_invert = midas_pred * s + t
    midas_aligned = 1 / (midas_aligned_invert + 10e-6)

    print(f"align_midas_withzeros ground_truth max: {ground_truth.max()} | min: {ground_truth.min()}")
    print(f"align_midas_withzeros midas_aligned max: {midas_aligned.max()} | min: {midas_aligned.min()}")

    return midas_aligned


def HWC3(x):

    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def reshape_to_shape(image, H, W, is_depth=False):
    if is_depth:
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4 if W > image.shape[1] > 1 else cv2.INTER_AREA)


def crop_to_shape(image, H, W):
    H_original, W_original = image.shape[:2]
    H_crop = H_original - H
    W_crop = W_original - W

    assert (H_crop >= 0)
    assert (W_crop >= 0)
    H_crop_half = int(H_crop // 2)
    W_crop_half = int(W_crop // 2)

    return image[H_crop_half : int(H_original - H_crop + H_crop_half),
                 W_crop_half : int(W_original - W_crop + W_crop_half)]


def resize_image(input_image, resolution=None, is_depth=False, ref_width=True, round=True, crop=False,
                 scaling_factor=None):
    if len(input_image.shape) == 3:
        H, W, C = input_image.shape
    else:
        H, W = input_image.shape
    H = float(H)
    W = float(W)

    if crop:
        H_crop = int(H) % 64
        W_crop = int(W) % 64
        print(f"before 64 crop: {input_image.shape}")
        input_image = input_image[int(H_crop / 2):int(H - H_crop / 2), int(W_crop / 2):int(W - W_crop / 2)]

        H = input_image.shape[0]
        W = input_image.shape[1]
        H = float(H)
        W = float(W)

        print(f"after 64 crop: {input_image.shape}")

    if scaling_factor == None:
        if ref_width:
            print(f"ref width")
            k = float(resolution) / W
        else:
            print(f"ref hight")
            k = float(resolution) / H
    else:
        k = scaling_factor

    print(f"k for resize: {k}")
    H *= k
    W *= k
    if round:
        print(f"will round")

        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
    else:
        print(f"wont round")
        H = int(H)
        W = int(W)
    print(f"before resize: {input_image.shape}")
    print(f"resolution: {resolution}")
    if is_depth:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    print(f"after resize: {img.shape}")
    return img


# for nicer looking plot titles
def break_up_string(text, line_limit=50):
    char_count = 0
    new_text = ""
    for word in text.split():
        if not new_text:
            new_text = word
            char_count = len(word)
        elif len(word) + char_count < line_limit:
            new_text = " ".join([new_text, word])
            char_count += len(word)
        else:
            new_text = "\n".join([new_text, word])
            char_count = len(word)
    return new_text


def get_sizing_params(resolution, original_shape, round_down=False):
    H_original = original_shape[0]
    W_original = original_shape[1]

    scaling_factor = float(resolution) / W_original

    H_resize = H_original * scaling_factor
    W_resize = W_original * scaling_factor
    if round_down:
        H_resize = int(H_resize // 64.0) * 64
        W_resize = int(W_resize // 64.0) * 64
    else:
        H_resize = int(np.round(H_resize / 64.0)) * 64
        W_resize = int(np.round(W_resize / 64.0)) * 64

    H_upsample = int(H_resize / scaling_factor)
    W_upsample = int(W_resize / scaling_factor)

    print(f"H_original: {H_original}")
    print(f"W_original: {W_original}")
    print(f"scaling_factor: {scaling_factor}")
    print(f"H_resize: {H_resize}")
    print(f"W_resize: {W_resize}")
    print(f"H_upsample: {H_upsample}")
    print(f"W_upsample: {W_upsample}")

    return H_original, W_original, H_resize, W_resize, scaling_factor, H_upsample, W_upsample


def prepare_nyu_controlnet_depth(x, scaling_factor=None):
    new_img = x.astype(np.float32)

    nonzero = np.nonzero(new_img)
    new_img[nonzero] -= np.min(new_img[nonzero])
    new_img /= np.max(new_img)
    new_img = (new_img * 255.0).clip(0, 255).astype(np.uint8)
    new_img = HWC3(new_img)

    if scaling_factor is not None:
        new_img = resize_image(new_img, is_depth=True, round=True, crop=False, scaling_factor=scaling_factor)
        H, W = new_img.shape[:2]
    else:
        H, W = new_img.shape[:2]

    control_image = Image.fromarray(np.uint8(new_img))

    control_image.show()

    return control_image, H, W


def prepare_nyu_controlnet_seg_nyu40(x, num_classes=50, image_resolution=512):
    color_group = np.stack([np.random.choice(range(256), size=3) for _ in range(num_classes)], axis=0)

    rgb_color_image = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32)
    rgb_color_image = color_group[np.uint16(x)]
    rgb_color_image = rgb_color_image.clip(0, 255).astype(np.uint8)
    rgb_color_image = HWC3(rgb_color_image)

    temp_img = resize_image(rgb_color_image, image_resolution)
    H, W = temp_img.shape[:2]
    new_img = cv2.resize(rgb_color_image, (W, H), interpolation=cv2.INTER_NEAREST)

    control_image = Image.fromarray(np.uint8(new_img))

    return control_image, H, W


# ---------------------


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


def load_nyu_rgbd(rgb_filename, d_filename):
    color_raw = mpimg.imread(rgb_filename)
    depth_raw = read_nyu_pgm(d_filename)
    color = o3d.geometry.Image(color_raw)
    depth = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth, convert_rgb_to_intensity=False)
    return rgbd_image


def resize(value, img):
    img = img.resize((value, value), Image.Resampling.LANCZOS)
    return img


def prepare_from_path(img_path):
    img = Image.open(img_path)
    img = resize(512, img)
    return img
