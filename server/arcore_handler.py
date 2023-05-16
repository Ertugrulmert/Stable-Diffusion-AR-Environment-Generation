import numpy as np
import cv2
import os, sys, io, time
import matplotlib.pyplot as plt
from utils.preprocessing import *
from models.controlnet_model_wrapper import ControlNetModelWrapper
from models_3d import point_clouds
import pymeshlab

parent_dir = os.path.dirname(os.path.dirname(__file__))


class ARCoreHandler:

    def __init__(self, mode="ground",
                 data_root='server/user_data',
                 cache_dir="",
                 only_ground=True):
        self.data_root = data_root

        self.model = ControlNetModelWrapper(condition_type="depth", multi_condition=False, only_ground=only_ground,
                                            result_root=data_root,
                                            cache_dir=cache_dir)

    def process_arcore_ground(self, rgb_filepath, depth_filepath, cam_rotation, confidence_filepath=None,  i=0,
                              resolution=512):
        predict_ground_depth_path = os.path.join(self.data_root, "ControlNet", "predicted_ground_truth_depth_maps",
                                                 f"{i}_predict_ground_depth")

        ground_pcd_path = os.path.join(self.data_root, "ground_point_clouds", f"{i}_arcore_ground_pcd.ply")

        rgb_image, depth_map = prepare_arcore_data(rgb_filepath, depth_filepath, confidence_filepath,
                                                   image_resolution=resolution, crop_rate=0.2)

        predict_ground_depth_map = self.model.infer_depth_map(rgb_image, save_name=predict_ground_depth_path,
                                                              display=False)

        predict_ground_depth_map = resize_image(predict_ground_depth_map, resolution=resolution)
        predict_ground_depth_map_aligned = align_midas_withzeros(predict_ground_depth_map, depth_map)

        c1, c2 = predict_ground_depth_map_aligned.shape
        center_depth = predict_ground_depth_map_aligned[c1//2, c2//2]

        original_pcd = point_clouds.get_point_cloud(rgb_image, predict_ground_depth_map_aligned,
                                                    pcd_path=ground_pcd_path,
                                                    display=False)

        #small_pcd = original_pcd.voxel_down_sample(voxel_size=0.0007)

        #radii = [0.001, 0.005, 0.01]
        #bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #    small_pcd, o3d.utility.DoubleVector(radii))

        #bpa_mesh = bpa_mesh.simplify_quadric_decimation(500)
        #bpa_mesh.remove_degenerate_triangles()
        #bpa_mesh.remove_duplicated_triangles()
        #bpa_mesh.remove_duplicated_vertices()
        #bpa_mesh.remove_non_manifold_edges()
        #bpa_mesh.scale(10, center=bpa_mesh.get_center())
        #T = np.eye(4)
        #T[0, 0] = -1
        #T[2, 2] = -1
        #bpa_mesh.transform(T)


        relative_mesh_path_obj = os.path.join("processed_user_data", f"{i}_mesh.obj")
        relative_mesh_path_ply = os.path.join("processed_user_data", f"{i}_mesh.ply")
        relative_material_path = os.path.join("processed_user_data", f"{i}_mesh.mtl")

        relative_mesh_path = os.path.join("processed_user_data", f"{i}_mesh.obj")
        mesh_name = f"{i}_mesh.obj"
        material_name = f"{i}_mesh.obj.mtl"
        texture_name = f"{i}_mesh.png"

        full_mesh_path_obj = os.path.join(self.data_root, f"{relative_mesh_path_obj}")
        #full_mesh_path_ply = os.path.join(self.data_root, f"{relative_mesh_path_ply}")

        #o3d.io.write_triangle_mesh(full_mesh_path_ply, bpa_mesh)

        #ms = pymeshlab.MeshSet()
        #ms.load_new_mesh(full_mesh_path_ply)
        #ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=1000, preservenormal=True)
        #ms.compute_color_transfer_vertex_to_face()
        #ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=200)
        #ms.compute_texmap_from_color(textname=f"{i}_mesh", textw=512, texth=512)
        #ms.save_current_mesh(full_mesh_path_obj)


        # -- NEW CODE
        self.process_mesh_marching_cubes(ground_pcd_path, full_mesh_path_obj, i, center_depth/10, cam_rotation)


        # -----------
        #relative_mesh_path = os.path.join("processed_user_data", "andy.obj")

        return mesh_name, material_name, texture_name


    def process_mesh_ball_pivoting(self, ground_pcd_path,full_mesh_path_obj,i):
        ms = pymeshlab.MeshSet()
        ms.clear()
        ms.load_new_mesh(ground_pcd_path)
        ms.apply_filter("apply_matrix_flip_or_swap_axis", flipx=True)
        ms.apply_filter("compute_matrix_from_rotation", rotaxis="Y axis", angle=180)
        ms.apply_filter("compute_normal_for_point_clouds", smoothiter=2, flipflag=True, k=100)
        ms.apply_filter("generate_simplified_point_cloud", samplenum=10000)
        ms.set_mesh_visibility(0, False)
        ms.apply_filter("delete_non_visible_meshes")
        ms.apply_filter("generate_surface_reconstruction_ball_pivoting")
        ms.apply_filter("delete_non_visible_meshes")
        ms.compute_color_transfer_vertex_to_face()
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        #ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")
        ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture', targetfacenum=1000,qualitythr=1, planarquadric=True)
        ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")
        ms.apply_filter("meshing_repair_non_manifold_edges")
        ms.apply_filter("meshing_close_holes", maxholesize=10, newfaceselected=False)
        ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=20)

        ms.compute_texmap_from_color(textname=f"{i}_mesh", textw=16, texth=16)
        ms.save_current_mesh(full_mesh_path_obj)

    def process_mesh_marching_cubes(self, ground_pcd_path, full_mesh_path_obj,i, center_depth, cam_rotation):
        ms = pymeshlab.MeshSet()
        ms.clear()
        ms.load_new_mesh(ground_pcd_path)

        print(f"center_depth {center_depth}")

        ms.apply_filter("compute_matrix_from_translation",axisz=center_depth )

        #if cam_rotation is not None:

            #x_angle = cam_rotation[0]
            #x_angle = 360 - x_angle if x_angle > 180 else x_angle
            #x_angle = 90 - x_angle


            #ms.apply_filter("compute_matrix_from_rotation", rotaxis="X axis", angle=x_angle)
            #ms.apply_filter("compute_matrix_from_rotation", rotaxis="Y axis", angle=cam_rotation[1])
            #ms.apply_filter("compute_matrix_from_rotation", rotaxis="Z axis", angle=-1*cam_rotation[2])

        ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=10)
        ms.apply_filter("apply_matrix_flip_or_swap_axis", flipx=True)
        ms.apply_filter("compute_matrix_from_rotation", rotaxis="Y axis", angle=180)

        start = time.time()
        ms.apply_filter("compute_normal_for_point_clouds", flipflag=True, k=100)
        end = time.time()
        print(f"compute_normal_for_point_clouds | time: {end - start}")

        start = time.time()
        ms.apply_filter("generate_simplified_point_cloud", samplenum=40000)
        end = time.time()
        print(f"generate_simplified_point_cloud | time: {end - start}")


        ms.set_mesh_visibility(0, False)
        #ms.apply_filter("delete_non_visible_meshes")

        start = time.time()
        ms.apply_filter("generate_marching_cubes_apss" )#, filterscale=4, resolution=100)
        end = time.time()
        print(f"generate_marching_cubes_apss | time: {end - start}")

        ms.apply_filter("meshing_poly_to_tri")

        start = time.time()
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=10000, preservenormal=True,
                        planarquadric=True, qualitythr=0.7)
        end = time.time()
        print(f"meshing_decimation_quadric_edge_collapse | time: {end - start}")

        #ms.apply_filter("delete_non_visible_meshes")
        #ms.compute_color_transfer_vertex_to_face()
        #ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")

        start = time.time()
        # Adding color
        ms.apply_filter("transfer_attributes_per_vertex", sourcemesh=0, targetmesh=ms.current_mesh_id())
        end = time.time()
        print(f"transfer_attributes_per_vertex | time: {end - start}")

        #ms.apply_filter("delete_non_visible_meshes")

        #start = time.time()
        #ms.apply_filter("meshing_repair_non_manifold_edges")
        #end = time.time()
        #print(f"meshing_repair_non_manifold_edges | time: {end - start}")

        #start = time.time()
        #ms.apply_filter("meshing_close_holes", maxholesize=10, newfaceselected=False)
        #end = time.time()
        #print(f"meshing_close_holes | time: {end - start}")

        #ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")

        #ms.compute_color_transfer_vertex_to_face()
        # ms.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
        #ms.apply_filter("apply_texmap_defragmentation", timelimit=5)
        #ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        ms.apply_filter("compute_texcoord_parametrization_flat_plane_per_wedge")
        ms.compute_texmap_from_color(textname=f"{i}_mesh" )#, textw=256, texth=256)

        ms.apply_filter("apply_normal_normalization_per_vertex")
        ms.save_current_mesh(full_mesh_path_obj)


    def get_serialized_object(self, path):
        #mesh = o3d.io.read_triangle_mesh(mesh_path)

        full_path = os.path.join(self.data_root, "processed_user_data", path)

        if os.path.isfile(full_path):

            with open(full_path, "rb") as fh:
                buf = io.BytesIO(fh.read())

                return buf
        else:
            return None


