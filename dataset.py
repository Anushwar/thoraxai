# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from config import LABEL_MAP


class NIH_Chest_Xray_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        train_test_file,
        transform=None,
        max_samples=None,
        include_no_finding=True,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.include_no_finding = include_no_finding

        df = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"))
        train_test_img_list = set([line.rstrip() for line in open(train_test_file)])

        for _, row in df.iterrows():
            img_name = row["Image Index"]
            if img_name in train_test_img_list:
                label = row["Finding Labels"].split("|")

                if not include_no_finding and "No Finding" in label:
                    continue

                for i in range(1, 13):
                    img_path = os.path.join(
                        data_dir, f"images_{i:03}/images/", img_name
                    )
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        break

                if max_samples is not None and len(self.image_paths) >= max_samples:
                    break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        num_labels = len(LABEL_MAP)
        binary_label = torch.zeros(num_labels)

        if not self.include_no_finding or "No Finding" not in label:
            for l in label:
                if l in LABEL_MAP:
                    binary_label[LABEL_MAP[l]] = 1

        return img, binary_label
