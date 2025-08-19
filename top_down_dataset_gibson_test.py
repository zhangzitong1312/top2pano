import json
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    top_down_cache = dict()

    def __init__(self):
        self.data = []
        for dir in os.listdir('./training/top_down_dataset_test'):
            if os.path.isdir('./training/top_down_dataset_test/' + dir):
                open_file = open('./training/top_down_dataset_test/' + dir + f'/{dir}_data.json')
                data = json.load(open_file)
                for camera in data["cameras"]:
                    camera_data = {
                        "source": './training/top_down_dataset_test/' + dir + "/" + data[
                            "mesh"] + f"_floor{camera['floor']}.png",
                        "target": './training/top_down_dataset_test/' + dir + f"/pano_rgb/{camera['name']}.png",
                        "depth": './training/top_down_dataset_test/' + dir + f"/pano_depth/{camera['name']}.png",
                        "segment": './training/top_down_dataset_test/' + dir + "/" + data[
                            "mesh"] + f"_floor{camera['floor']}_seg.png",
                        "prompt": "Using an indoor top-down view image and its corresponding segmentation as the condition input, learn the complete geometric structure of the room based on this view to generate a voxel representation of the space. The segmentation helps distinguish between the floor and furniture, enabling the model to capture the overall room layout, as well as the shape, position, and spatial relationships of objects and walls. By leveraging both the visual and structural cues from the segmentation, the model can achieve a more precise reconstruction of the room’s geometric structure from the top-down perspective, ensuring accurate height estimation and detailed object representation.",
                        "render": "Generate an ultra-high-definition panoramic indoor scene using the provided condition input. The condition consists of a composite image: a coarse RGB image and a corresponding coarse depth map obtained from density estimation. Use the coarse RGB image to restore the floor color, furniture positions, and their appearance, while utilizing the coarse depth map to reconstruct the structural layout of the room. Ensure that the outpainted regions blend seamlessly with the coarse RGB input, accurately reflect the furniture details, and align with the spatial cues provided by the coarse depth. The final result should be a coherent, high-quality panoramic RGB image that fully restores the indoor environment with realistic textures and structural integrity.",
                        #"render": "[Chinese style] Indoor Scene",
                        "location": camera["position"],
                        "rotation": camera["rotation"],
                        "screen_location": camera["screen_position"],
                        "pixel_scale": data["pixel_scale"]
                    }
                    self.data.append(camera_data)
        json.dump(self.data, open('./training/top_down_dataset_test/data.json', 'w'))

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
        segmentation = item['segment']
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
                    ground_truth_depth=ground_truth_depth, black_mask=source_mask,
                    top_down_image=cv2.imread(source_filename, cv2.IMREAD_COLOR), dataset_name = 'gibson')
