import json
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip

from utils.constants import CONCEPT_TO_IDX


class LilacDataset(Dataset):
    def __init__(self, cfg, split, mode="train", preprocessor=None):
        self.root_dir = os.path.join(cfg.data_dir, cfg.dataset)
        self.concept2idx = CONCEPT_TO_IDX[cfg.dataset.split("-")[-1]]
        with open(os.path.join(self.root_dir, f"{mode}.json")) as file:
            self.data = [d for d in json.load(file) if d["task"] in split["indices"]]
        self.img_dir = os.path.join(self.root_dir, "images", mode)
        self.preprocess = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        mission = clip.tokenize(sample["mission"])
        concept = torch.tensor(self.concept2idx[sample["concept"]])
        task = torch.tensor(sample["task"])

        images = []
        for img_name in sample["images"]:
            img_path = os.path.join(self.img_dir, img_name)
            image = self.preprocess(Image.open(img_path).convert("RGB"))
            images.append(image)

        return (
            images[0],
            images[1],
            images[2],
            mission,
            concept,
            task,
        )


class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.num_seen_samples = 0
        self.losses = []

    def __getitem__(self, idx):
        return tuple(self.data[idx])

    def __len__(self):
        return len(self.data)

    def reservoir_sampling(self, batch):
        batch = [b.cpu() for b in batch]
        for i in range(len(batch[5])):
            if len(self.data) < self.capacity:
                self.data.append([b[i] for b in batch])
            else:
                pos = random.randint(0, self.num_seen_samples)
                if pos < self.capacity:
                    self.data[pos] = [b[i] for b in batch]
            self.num_seen_samples += 1
