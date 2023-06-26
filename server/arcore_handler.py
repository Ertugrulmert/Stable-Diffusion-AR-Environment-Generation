import io
from utils.preprocessing import *
from models.controlnet_model_wrapper import ControlNetModelWrapper
from models_3d import point_clouds, mesh_processing

parent_dir = os.path.dirname(os.path.dirname(__file__))


class ARCoreHandler:

    def __init__(self, mode="ground",
                 data_root='server/user_data',
                 resolution=384,
                 num_steps=20,
                 condition_type="depth",
                 multi_condition=False,
                 cache_dir="",
                 only_ground=True):
        self.data_root = data_root

        self.model = ControlNetModelWrapper(condition_type=condition_type, multi_condition=multi_condition, only_ground=only_ground,
                                            result_root=data_root,
                                            resolution=resolution,
                                            num_steps=num_steps,
                                            cache_dir=cache_dir)

    def process_arcore_ground(self, rgb_filepath, depth_filepath, cam_rotation, i=0,
                              resolution=384):
        mesh_name = f"{i}_mesh.obj"
        material_name = f"{i}_mesh.obj.mtl"
        texture_name = f"{i}_mesh.png"
        relative_mesh_path_obj = os.path.join("processed_user_data", f"{i}_mesh.obj")
        full_mesh_path_obj = os.path.join(self.data_root, f"{relative_mesh_path_obj}")

        predict_ground_depth_path = os.path.join(self.data_root, "ControlNet", "predicted_ground_truth_depth_maps",
                                                 f"{i}_predict_ground_depth")

        ground_pcd_path = os.path.join(self.data_root, "ground_point_clouds", f"{i}_arcore_ground_pcd.ply")

        rgb_image, depth_map = prepare_arcore_data(rgb_filepath, depth_filepath,
                                                   image_resolution=resolution, crop_rate=0.2)

        predict_ground_depth_map = self.model.infer_depth_map(rgb_image, save_name=predict_ground_depth_path,
                                                              display=False)

        predict_ground_depth_map = resize_image(predict_ground_depth_map, resolution=resolution)
        predict_ground_depth_map_aligned = align_midas_withzeros(predict_ground_depth_map, depth_map)

        c1, c2 = predict_ground_depth_map_aligned.shape
        center_depth = predict_ground_depth_map_aligned[c1 // 2, c2 // 2]

        original_pcd = point_clouds.get_point_cloud(rgb_image, predict_ground_depth_map_aligned,
                                                    pcd_path=ground_pcd_path,
                                                    display=False)
        # -- NEW CODE
        mesh_processing.process_mesh_marching_cubes(ground_pcd_path, full_mesh_path_obj, i, center_depth / 10,
                                                    cam_rotation)

        return mesh_name, material_name, texture_name

    def process_arcore_generative(self, rgb_filepath, depth_filepath, cam_rotation, i=0, camIntrinsics="",
                                  only_ground=False, prompt=""):
        mesh_name = f"{i}_mesh.obj"
        material_name = f"{i}_mesh.obj.mtl"
        texture_name = f"{i}_mesh.png"
        relative_mesh_path_obj = os.path.join("processed_user_data", f"{i}_mesh.obj")
        full_mesh_path_obj = os.path.join(self.data_root, f"{relative_mesh_path_obj}")

        pcd_path, center_depth = self.model.run_ARCore_pipeline(rgb_filepath, depth_filepath, i=i, prompt=prompt,
                                                                camIntrinsics=camIntrinsics,
                                                                only_ground=only_ground, display=False, save_eval=True)

        # -- NEW CODE
        mesh_processing.process_mesh_marching_cubes(pcd_path, full_mesh_path_obj, i, center_depth / 10,
                                                    cam_rotation)

        return mesh_name, material_name, texture_name

    def get_serialized_object(self, path):
        # mesh = o3d.io.read_triangle_mesh(mesh_path)

        full_path = os.path.join(self.data_root, "processed_user_data", path)

        if os.path.isfile(full_path):

            with open(full_path, "rb") as fh:
                buf = io.BytesIO(fh.read())

                return buf
        else:
            return None
