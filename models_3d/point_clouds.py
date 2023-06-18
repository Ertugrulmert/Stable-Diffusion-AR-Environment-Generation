import os, sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from PIL import Image
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import json

import matplotlib as mpl
import matplotlib.cm as cm

from utils.preprocessing import *


def get_point_cloud(rgb_image, depth_image, camIntrinsics="", pcd_path="", display=False):

    print("get point cloud")
    print(f"rgb shape {rgb_image.shape}")
    print(f"depth shape {depth_image.shape}")

    new_depth_image = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image), o3d.geometry.Image(new_depth_image),
        depth_scale=1, convert_rgb_to_intensity=False)

    if camIntrinsics is not None and camIntrinsics:
        print(f"process intrinsic here {camIntrinsics}")
    """
    2023 / 06 / 18
    12: 56:05.716
    14469
    29876
    Info
    Unity
    Camera
    intrinsics
    {'focalLength.x': 482.2705, 'focalLength.y': 483.8367, 'principalPoint.x': 320.5844, 'principalPoint.y': 236.8517,
     'resolution.x': 640, 'resolution.y': 480}"""

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        #o3d.camera.PinholeCameraIntrinsic(
        #    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        o3d.camera.PinholeCameraIntrinsic(480, 640, 483.8367, 482.2705, 236.8517, 320.5844))

    #o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0104, max_nn=12))



    # Flip it, otherwise the point cloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # writing point cloud to file
    if pcd_path:
        o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=False, compressed=False, print_progress=False)

    if display:
        o3d.visualization.draw_geometries([pcd])

    return pcd


def rebuild_point_clouds(rgb_image, depth_map, i, result_root='./results/NYU/', display=True, condition_type="seg"):
    if condition_type:
        condition_id = condition_type if isinstance(condition_type, str) else f"_{'_'.join(condition_type)}_"

        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth_from_{condition_id}.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated_from_{condition_id}.png"
        gen_pcd_path = result_root + f"ControlNet/gen_point_clouds/{i}_gen_pcd_from_{condition_id}"
    else:
        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated.png"
        gen_pcd_path = result_root + f"ControlNet/gen_point_clouds/{i}_gen_pcd"

    ground_pcd_path = result_root + f"ground_point_clouds/{i}_ground_pcd"

    predict_depth_map = np.load(predict_depth_path)
    image_resolution = predict_depth_map.shape[0]

    src_img_np, ground_depth_map, original_image_W = prepare_nyu_data(rgb_image, depth_map, image_resolution=image_resolution)
    gen_img = Image.open(gen_img_path)
    gen_img.show()
    gen_img_np = np.array(gen_img)

    predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

    original_pcd = get_point_cloud(src_img_np, ground_depth_map, pcd_path=ground_pcd_path + ".pcd",
                                   display=display)
    generated_pcd = get_point_cloud(gen_img_np, predict_depth_map_aligned, pcd_path=gen_pcd_path + ".pcd",
                                    display=display)

    # pcd = o3d.io.read_point_cloud( nyu_result_root + f"point_clouds/{i}_ground_pcd.pcd")
    # o3d.visualization.draw_geometries([pcd])

    return original_pcd, generated_pcd


def rebuild_point_clouds_ground_depth(rgb_image, depth_map, i, result_root='./results/NYU/',
                                      display=True, condition_type="seg"):

    if condition_type:
        condition_id = condition_type if isinstance(condition_type, str) else f"_{'_'.join(condition_type)}_"

        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth_from_{condition_id}.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated_from_{condition_id}.png"
        gen_pcd_path = result_root + f"ControlNet/gen_point_clouds/{i}_gen_pcd_from_{condition_id}"
    else:
        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated.png"
        gen_pcd_path = result_root + f"ControlNet/gen_point_clouds/{i}_gen_pcd"

    ground_pcd_path = result_root + f"ground_point_clouds/{i}_ground_pcd"

    predict_depth_map = np.load(predict_depth_path)
    image_resolution = predict_depth_map.shape[0]

    src_img_np, ground_depth_map, original_image_W = prepare_nyu_data(rgb_image, depth_map, image_resolution=image_resolution)
    #resized_src_img = resize_image(src_img_np, image_resolution)
    gen_img = Image.open(gen_img_path)
    gen_img.show()
    gen_img_np = np.array(gen_img)

    #groundmap_for_heatmap = resize_image(ground_depth_map, image_resolution)
    #predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

    original_pcd = get_point_cloud(src_img_np, ground_depth_map, pcd_path=ground_pcd_path + ".pcd",
                                   display=display)
    generated_pcd = get_point_cloud(gen_img_np, ground_depth_map, pcd_path=gen_pcd_path + "_ground_depth.pcd",
                                    display=display)

    # pcd = o3d.io.read_point_cloud( nyu_result_root + f"point_clouds/{i}_ground_pcd.pcd")
    # o3d.visualization.draw_geometries([pcd])

    return original_pcd, generated_pcd


def rebuild_point_clouds_heatmap(rgb_image, depth_map, i, result_root='./results/NYU/', display=False,
                                 condition_type="seg"):

    if condition_type:
        condition_id = condition_type if isinstance(condition_type, str) else f"_{'_'.join(condition_type)}_"
        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth_from_{condition_id}.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated_from_{condition_id}.png"
    else:
        predict_depth_path = result_root + f"ControlNet/gen_depth_maps/{i}_gen_depth.npy"
        gen_img_path = result_root + f"ControlNet/2d_images/{i}_generated.png"

    ground_pcd_path = result_root + f"ground_point_clouds/{i}_ground_pcd"

    predict_depth_map = np.load(predict_depth_path)
    image_resolution = predict_depth_map.shape[0]

    src_img_np, ground_depth_map, original_image_W = prepare_nyu_data(rgb_image, depth_map, image_resolution=image_resolution)

    gen_img = Image.open(gen_img_path)
    gen_img.show()

    predict_depth_map_aligned = align_midas(predict_depth_map, ground_depth_map)

    difference_map = ground_depth_map - predict_depth_map_aligned

    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    cmap = plt.get_cmap('RdBu_r')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    heatmap = (m.to_rgba(difference_map)[:, :, :3] * 256).astype(np.uint8)

    gen_pcd_path = result_root + f"ControlNet/gen_point_clouds/{i}_heatmap_pcd.pcd"

    pcd = get_point_cloud(heatmap, predict_depth_map_aligned, pcd_path=gen_pcd_path, display=display)

    # pcd = o3d.io.read_point_cloud( nyu_result_root + f"point_clouds/{i}_heatmap_pcd.pcd")
    if display:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def get_mesh_from_pcd(pcd, method="ball_rolling"):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

        if method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                                        depth=10, width=0, scale=1.1,
                                                                                        linear_fit=True)
            print(mesh)
            o3d.visualization.draw_geometries([mesh])

            return mesh, densities

        if method == "ball_pivoting":
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            o3d.visualization.draw_geometries([pcd, mesh])

            return mesh

        if method == "alpha":
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

            for alpha in np.logspace(np.log10(0.5), np.log10(0.1), num=3):
                print(f"alpha={alpha:.3f}")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map)
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        if method == "ball_rolling":
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([radius, radius * 2]))

            dec_mesh = mesh.simplify_quadric_decimation(100000)
            dec_mesh.remove_degenerate_triangles()
            dec_mesh.remove_duplicated_triangles()
            dec_mesh.remove_duplicated_vertices()
            dec_mesh.remove_non_manifold_edges()

            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
