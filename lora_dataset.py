import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.source_dir = os.path.join(data_root, "source")
        self.target_dir = os.path.join(data_root, "target")

        self.file_list = [
            fname for fname in os.listdir(self.target_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]

        prompt = "[Chinese style] Indoor Scene"
        prompt = "[European style] Indoor Scene"
        prompt = "[Indian style] Indoor Scene"
        prompt = "[African style] Indoor Scene"
        prompt = "[Middle East style] Indoor Scene"
        source_path = os.path.join(self.source_dir, base_name)
        target_path = os.path.join(self.target_dir, base_name)

        #source = cv2.imread(source_path)
        source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        source = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)


        target = cv2.imread(target_path)
        target = cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if source is None:
            raise FileNotFoundError(f"Source image missing: {source_path}")
        if target is None:
            raise FileNotFoundError(f"Target image missing: {target_path}")


        source = source.astype(np.float32) / 255.0  # [0, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0  # [-1, 1]

        return dict(jpg=target, txt=prompt, hint=source, ground_truth=source)

