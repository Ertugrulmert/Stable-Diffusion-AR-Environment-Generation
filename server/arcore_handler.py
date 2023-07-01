import io
from utils.preprocessing import *
from models.controlnet_model_wrapper import ControlNetModelWrapper
from models_3d import point_clouds, mesh_processing

parent_dir = os.path.dirname(os.path.dirname(__file__))


class ARCoreHandler:

    def __init__(self,
                 data_root='server/user_data',
                 resolution=384,
                 num_steps=20,
                 condition_type="depth",
                 multi_condition=False,
                 cache_dir="",
                 only_ground=True):
        self.data_root = data_root

        self.model = ControlNetModelWrapper(condition_type=condition_type, multi_condition=multi_condition,
                                            only_ground=only_ground,
                                            result_root=data_root,
                                            resolution=resolution,
                                            num_steps=num_steps,
                                            cache_dir=cache_dir)

    def process_arcore_generative(self, rgb_filepath, depth_filepath, cam_rotation, i=0, camIntrinsics="",
                                  only_ground=False, prompt=""):

        return self.model.run_ARCore_pipeline(rgb_filepath, depth_filepath, i=i, prompt=prompt,
                                              camIntrinsics=camIntrinsics,
                                              only_ground=only_ground, display=False, save_eval=True)

    def get_serialized_object(self, path):

        full_path = os.path.join(self.data_root, "processed_user_data", path)

        if os.path.isfile(full_path):

            with open(full_path, "rb") as fh:
                buf = io.BytesIO(fh.read())

                return buf
        else:
            return None
