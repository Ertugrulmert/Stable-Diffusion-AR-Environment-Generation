from PIL import Image
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os
import json


class Visualiser:

    def get_depth_heat_map(ground_depth_map, predict_ground_depth_map, predict_depth_map, img_id=None,
                           save_name='', display=False):

        ground_predict_ground_diff = ground_depth_map - predict_ground_depth_map
        predict_ground_gen_diff = predict_ground_depth_map - predict_depth_map
        ground_gen_diff = ground_depth_map - predict_depth_map

        _min, _max = np.amin(ground_depth_map), np.amax(ground_depth_map)

        fig, ax = plt.subplots(2, 3, figsize=(6*3 + 3, 6*3 + 3), layout='constrained')
        ax[0,0].imshow(ground_depth_map, vmin=_min, vmax=_max)
        ax[0,1].imshow(predict_ground_depth_map, vmin=_min, vmax=_max)
        row_0 = ax[0,2].imshow(predict_depth_map, vmin=_min, vmax=_max)

        ax[0,0].set_title('Ground Truth', fontsize=16)
        ax[0,1].set_title('Estimated Ground Truth', fontsize=16)
        ax[0,2].set_title('Generated', fontsize=16)

        cbar_0 = fig.colorbar(row_0, ax=ax[0,2], shrink=0.6)
        cbar_0.set_label('Depth Scale', rotation=90, labelpad=5)
        cbar_0.ax.set_yticklabels(["{:.2}".format(i) + " m" for i in cbar_0.get_ticks()])  # set ticks of your format


        heat_min = -3
        heat_max = 3

        ax[1,0].imshow(ground_predict_ground_diff, cmap='RdBu_r', vmin=heat_min, vmax=heat_max)
        ax[1,1].imshow(predict_ground_gen_diff, cmap='RdBu_r', vmin=heat_min, vmax=heat_max)
        row_1 = ax[1,2].imshow(ground_gen_diff, cmap='RdBu_r', vmin=heat_min, vmax=heat_max)

        cbar_1 = fig.colorbar(row_1, ax=ax[1,2], shrink=0.6)
        cbar_1.set_label('Ground truth - Predicted', rotation=90, labelpad=5)
        cbar_1.ax.set_yticklabels(["{:.2}".format(i) + " m" for i in cbar_1.get_ticks()])  # set ticks of your format

        ax[1,0].set_title('Ground Truth - Predicted Ground Truth', fontsize=16)
        ax[1,1].set_title('Predicted Ground Truth - Generated', fontsize=16)
        ax[1,2].set_title('Ground Truth - Generated', fontsize=16)

        for row in ax:
            for a in row:
                a.set_xticks([])
                a.set_yticks([])

        if img_id is not None:
            fig.suptitle(f"Depth Maps for Image - {img_id}", fontsize=18 )#, y=0.95)
        else:
            fig.suptitle(f"Depth Maps", fontsize=18 )#, y=0.95)

        if save_name:
            fig.savefig(save_name)
        if display:
            plt.show()

    def get_depth_heat_map_no_ground(predict_ground_depth_map, predict_depth_map, img_id=None, save_name=''):

        depth_diff = predict_ground_depth_map - predict_depth_map

        _min, _max = np.amin(predict_ground_depth_map), np.amax(predict_ground_depth_map)

        fig, ax = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
        ax[0].imshow(predict_ground_depth_map, vmin=_min, vmax=_max)
        ax[1].imshow(predict_depth_map, vmin=_min, vmax=_max)

        heat_min = -4
        heat_max = 4

        diff = ax[2].imshow(depth_diff, cmap='RdBu_r', vmin=heat_min, vmax=heat_max)

        cbar = fig.colorbar(diff, ax=ax[2], shrink=0.6)
        cbar.set_label('Predicted Ground Truth - Predicted Generated', rotation=90, labelpad=5)
        cbar.ax.set_yticklabels(["{:.2}".format(i) + " m" for i in cbar.get_ticks()])  # set ticks of your format

        ax[0].set_title('Predicted Ground Truth', fontsize=16)
        ax[1].set_title('Generated', fontsize=16)
        ax[2].set_title('Difference Heat Map', fontsize=16)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        if img_id is not None:
            fig.suptitle(f"Depth Maps for Image - {img_id}", fontsize=18, y=1.1)
        else:
            fig.suptitle(f"Depth Maps", fontsize=18, y=1.1)

        if save_name:
            fig.savefig(save_name)

        plt.show()

    def capture_pcd_with_view_params(pcd, pcd_path, view_setting_path):
        vis = o3d.visualization.Visualizer()

        vis.create_window(visible=False)
        vis.add_geometry(pcd)

        with open(view_setting_path, "r") as f:
            js = json.load(f)

        vc = vis.get_view_control()
        vc.change_field_of_view(js['trajectory'][0]['field_of_view'])
        vc.set_front(js['trajectory'][0]['front'])
        vc.set_lookat(js['trajectory'][0]['lookat'])
        vc.set_up(js['trajectory'][0]['up'])
        vc.set_zoom(js['trajectory'][0]['zoom'])

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(pcd_path)
        vis.destroy_window()

