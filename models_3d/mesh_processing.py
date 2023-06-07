import pymeshlab
import os, sys, io, time

def process_mesh_ball_pivoting(ground_pcd_path, full_mesh_path_obj, i):
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
    # ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")
    ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture', targetfacenum=1000, qualitythr=1,
                    planarquadric=True)
    ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_close_holes", maxholesize=10, newfaceselected=False)
    ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=20)

    ms.compute_texmap_from_color(textname=f"{i}_mesh", textw=16, texth=16)
    ms.save_current_mesh(full_mesh_path_obj)


def process_mesh_marching_cubes(ground_pcd_path, full_mesh_path_obj, i, center_depth, cam_rotation):
    ms = pymeshlab.MeshSet()
    ms.clear()
    ms.load_new_mesh(ground_pcd_path)

    print(f"center_depth {center_depth}")


    ms.apply_filter("compute_matrix_from_translation", traslmethod='Center on Scene BBox')

    #ms.apply_filter("compute_matrix_from_translation", axisx=0.5)

    ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=2)

    ms.apply_filter("compute_matrix_from_translation", axisz=center_depth*2)

    # if cam_rotation is not None:

    # x_angle = cam_rotation[0]
    # x_angle = 360 - x_angle if x_angle > 180 else x_angle
    # x_angle = 90 - x_angle

    # ms.apply_filter("compute_matrix_from_rotation", rotaxis="X axis", angle=x_angle)
    # ms.apply_filter("compute_matrix_from_rotation", rotaxis="Y axis", angle=cam_rotation[1])
    # ms.apply_filter("compute_matrix_from_rotation", rotaxis="Z axis", angle=-1*cam_rotation[2])

    #ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=10)
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
    # ms.apply_filter("delete_non_visible_meshes")

    start = time.time()
    ms.apply_filter("generate_marching_cubes_apss")  # , filterscale=4, resolution=100)
    end = time.time()
    print(f"generate_marching_cubes_apss | time: {end - start}")

    ms.apply_filter("meshing_poly_to_tri")

    start = time.time()
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=15000, preservenormal=True,
                    planarquadric=True, qualitythr=0.8)
    end = time.time()
    print(f"meshing_decimation_quadric_edge_collapse | time: {end - start}")

    # ms.apply_filter("delete_non_visible_meshes")
    # ms.compute_color_transfer_vertex_to_face()
    # ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")

    start = time.time()
    # Adding color
    ms.apply_filter("transfer_attributes_per_vertex", sourcemesh=0, targetmesh=ms.current_mesh_id())
    end = time.time()
    print(f"transfer_attributes_per_vertex | time: {end - start}")

    # ms.apply_filter("delete_non_visible_meshes")

    # start = time.time()
    # ms.apply_filter("meshing_repair_non_manifold_edges")
    # end = time.time()
    # print(f"meshing_repair_non_manifold_edges | time: {end - start}")

    # start = time.time()
    # ms.apply_filter("meshing_close_holes", maxholesize=10, newfaceselected=False)
    # end = time.time()
    # print(f"meshing_close_holes | time: {end - start}")

    # ms.apply_filter("apply_coord_laplacian_smoothing_surface_preserving")

    # ms.compute_color_transfer_vertex_to_face()
    # ms.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
    # ms.apply_filter("apply_texmap_defragmentation", timelimit=5)
    # ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    ms.apply_filter("compute_texcoord_parametrization_flat_plane_per_wedge")
    ms.compute_texmap_from_color(textname=f"{i}_mesh")  # , textw=256, texth=256)

    ms.apply_filter("apply_normal_normalization_per_vertex")
    ms.save_current_mesh(full_mesh_path_obj)
