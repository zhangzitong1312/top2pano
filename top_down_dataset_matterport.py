import json
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


dataset_path = './training/dataset_matterport/dataset_matterport'

class MyDataset(Dataset):
    top_down_cache = dict()


    def __init__(self):
        self.data = []
        open_file = open(os.path.join(dataset_path, "train_data.json"))
        data = json.load(open_file)
        for camera in data:
            camera_data = {
                "source": os.path.join(dataset_path, camera["top_down"]),
                "target": os.path.join(dataset_path, camera["panorama"]),
                "depth": os.path.join(dataset_path, (camera["panorama"].replace("/pano_rpg/", "/depth/")).replace("_pano.jpg", "_depth.png")),
                "prompt": "Using an indoor top-down view image and its corresponding segmentation as the condition input, learn the complete geometric structure of the room based on this view to generate a voxel representation of the space. The segmentation helps distinguish between the floor and furniture, enabling the model to capture the overall room layout, as well as the shape, position, and spatial relationships of objects and walls. By leveraging both the visual and structural cues from the segmentation, the model can achieve a more precise reconstruction of the room’s geometric structure from the top-down perspective, ensuring accurate height estimation and detailed object representation.",
                "render": "An ultra-high-definition panoramic indoor scene. Using the provided coarse depth map and coarse color image as conditions, please reconstruct the room's structure and generate both a high-quality panoramic RGB image and a refined depth map of the indoor space. Ensure that the generated outputs maintain the same structure as the coarse depth map, while preserving the correct colors of the floor and furniture.",
                "location": camera["location"],
                "rotation": camera["rotation"],
                "screen_location": camera["screen_location"],
                "pixel_scale": camera["pixel_scale"]
            }
            self.data.append(camera_data)
        # saving this for later
        json.dump(self.data, open(os.path.join(dataset_path, "data_processed_train.json"), 'w'))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        location = item['location']
        rotation = item['rotation']
        screen_location = item['screen_location']
        pixel_scale = item['pixel_scale']
        render_prompt = item['render']
        ground_truth_depth = item['depth']
        segmentation = source_filename.replace('.png', '_seg.png')

        source = cv2.imread(source_filename, cv2.IMREAD_COLOR)
        source = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)

        seg = cv2.imread(segmentation, cv2.IMREAD_COLOR)
        seg = cv2.resize(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)


        black_pixels = np.where((source[:, :, 0] <= 5) &
                                (source[:, :, 1] <= 5) &
                                (source[:, :, 2] <= 5))
        black_pixel_positions = list(zip(black_pixels[0], black_pixels[1]))


        source_mask = torch.zeros(512, 512, 1, dtype=torch.bool)
        for i in range(len(black_pixel_positions)):
            source_mask[black_pixel_positions[i][0], black_pixel_positions[i][1], :] = True
            source[black_pixel_positions[i][0], black_pixel_positions[i][1], :] = 255

        target = cv2.imread(target_filename)
        target = cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        ground_truth_depth = cv2.imread(ground_truth_depth)
        ground_truth_depth = cv2.resize(cv2.cvtColor(ground_truth_depth, cv2.COLOR_BGR2RGB), (512, 512),
                                        interpolation=cv2.INTER_AREA)
        ground_truth_depth = cv2.cvtColor(ground_truth_depth, cv2.COLOR_BGR2RGB)
        ground_truth_depth = (ground_truth_depth - ground_truth_depth.min()) / (
                    ground_truth_depth.max() - ground_truth_depth.min())
        ground_truth_depth = ground_truth_depth.astype(np.float32)
        detected_map = source.astype(np.float32) / 255.0
        seg = seg.astype(np.float32) / 255.0
        detected_map = np.concatenate((detected_map, seg), axis=2)
        noise_input = torch.rand((512, 512, 3)) * 2 - 1
        pixel_scale = pixel_scale / 2
        screen_location = [int(screen_location[0] / 2), int(screen_location[1] / 2)]
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0


        return dict(jpg=source, txt=prompt, hint=detected_map, location=location,
                    rotation=rotation, screen_location=screen_location, pixel_scale=pixel_scale,
                    ground_truth=target, render_prompt=render_prompt, noise_input=noise_input,
                    black_mask=source_mask, top_down_image=cv2.imread(source_filename, cv2.IMREAD_COLOR),
                    ground_truth_depth=ground_truth_depth, dataset_name='matterport')

