import json
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    top_down_cache = dict()

    def __init__(self, source_paths, screen_location_paths, pixel_scale):
        self.data = []
        screen_location = np.load(screen_location_paths)
        for idx in range(screen_location.shape[0]):
            pixel_scale = pixel_scale
            screen_location[idx][1] = 1024 - screen_location[idx][1]
            camera_location = [int(screen_location[idx][0] / pixel_scale), int(screen_location[idx][1] / pixel_scale), 1.5 ]

            camera_data = {
                "source": source_paths,
                "target": source_paths,  # Assuming target is same as source for now
                "prompt": "Using an indoor top-down view image as the condition input, learn the complete geometric structure of the room based on this view to generate a voxel representation of the space. Focus on capturing the height, shape, and detailed information of every object and wall within the room.",
                #"render": "Generate high-quality panoramic images of the room with the camera positioned at various angles. Use the provided coarse depth map and coarse color image to accurately reconstruct the indoor environment, capturing the complete structure, height, and detailed features of the walls, objects, and furniture within the space.",
                "render": "[Chinese-style] indoor room",
                #"render": "[Wall-Stone-style] indoor room",
                #"render": "[Japanese-style] indoor room",
                #"render": "[Wall-Red-style] indoor room",
                #"render": "[Medieval Europe style] indoor room",
                "location": camera_location,
                "screen_location": screen_location[idx],
                "pixel_scale": pixel_scale,
                "rotation": [0, 0, 0, 1],
            }
            self.data.append(camera_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        torch.manual_seed(42)

        item = self.data[idx]

        source_filename = item['source']
        prompt = item['prompt']
        location = item['location']
        screen_location = item['screen_location']
        pixel_scale = item['pixel_scale']
        render_prompt = item['render']
        rotation = item['rotation']

        source = cv2.imread(source_filename, cv2.IMREAD_COLOR)
        source = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)

        black_pixels = np.where((source[:, :, 0] <= 5) &
                                (source[:, :, 1] <= 5) &
                                (source[:, :, 2] <= 5))
        black_pixel_positions = list(zip(black_pixels[0], black_pixels[1]))

        source_mask = torch.zeros(512, 512, 1, dtype=torch.bool)
        for i in range(len(black_pixel_positions)):
            source_mask[black_pixel_positions[i][0], black_pixel_positions[i][1], :] = True
            source[black_pixel_positions[i][0], black_pixel_positions[i][1], :] = 255

        detected_map = source.astype(np.float32) / 255.0
        noise_input = torch.rand((512, 512, 3)) * 2 - 1
        #torch.save(noise_input, "noise_input.pth")

        source = (source.astype(np.float32) / 127.5) - 1.0

        pixel_scale = pixel_scale / 2
        screen_location = [int(screen_location[0] / 2), int(screen_location[1] / 2)]

        return dict(jpg=source, txt=prompt, hint=detected_map, location=location, ground_truth=noise_input, ground_truth_depth=source,
                    screen_location=screen_location, pixel_scale=pixel_scale,
                    render_prompt=render_prompt, noise_input=noise_input,
                    black_mask=source_mask, top_down_image=cv2.imread(source_filename, cv2.IMREAD_COLOR), rotation=rotation, dataset_name='gibson')


