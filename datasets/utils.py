import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from itertools import product

import numpy as np
from torch.utils.data import DataLoader
from omegaconf import open_dict
import clip

from utils.constants import *
from datasets import LilacDataset


def download_dataset(cfg):
    url = f"https://www2.informatik.uni-hamburg.de/wtm/datasets2/{cfg.dataset}.zip"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with ZipFile(BytesIO(response.content)) as zfile:
        for data in response.iter_content(1024):
            progress_bar.update(len(data))
        zfile.extractall(cfg.data_dir)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Something went wrong while downloading the dataset!")


def load_data(cfg):
    tasks = list(
        product(COLORS, OBJECT_TYPES, DIRECTIONS)
        if cfg.dataset.split("-")[-1] == "2d"
        else product(BLOCK_COLORS, SIZES, BOWL_COLORS)
    )
    
    with open_dict(cfg):
        cfg.all_tasks = {"indices": list(range(len(tasks))), "tasks": tasks}

    params = {
        "batch_size": cfg.batch_size,
        "drop_last": False,
        "num_workers": 8,
        "pin_memory": True,
        "shuffle": True,
    }

    # Create num_tasks dataloader for continual setup
    assert len(tasks) % cfg.T == 0
    if cfg.continual:
        # continual training
        permuted_task_indices = np.random.permutation(len(tasks)).reshape(cfg.T, -1)
    else:
        # joint training
        permuted_task_indices = np.random.permutation(len(tasks)).reshape(1, -1)

    # Import CLIP preprocessor
    if cfg.arch == "FiLM_ResNet" or cfg.arch == "Transformer_ResNet":
        _, preprocess = clip.load("RN50")
    else:
        _, preprocess = clip.load("ViT-B/16")

    with open_dict(cfg):
        cfg.continual_tasks = {"indices": [], "tasks": []}
    continual_dl = []
    eval_dl = []
    test_dl = []

    for indices in permuted_task_indices:
        split = {
            "indices": [
                j for i, j in enumerate(cfg.all_tasks["indices"]) if i in indices
            ],
            "tasks": [
                j for i, j in enumerate(cfg.all_tasks["tasks"]) if i in indices
            ],
        }
        with open_dict(cfg):
            cfg.continual_tasks["indices"].append(split["indices"])
            cfg.continual_tasks["tasks"].append(split["tasks"])

        continual_dl.append(
            DataLoader(
                dataset=LilacDataset(cfg, split=split, mode="train", preprocessor=preprocess), **params
            )
        )
        eval_dl.append(
            DataLoader(
                dataset=LilacDataset(cfg, split=split, mode="val", preprocessor=preprocess), **params
            )
        )
        test_dl.append(
            DataLoader(
                dataset=LilacDataset(cfg, split=split, mode="test", preprocessor=preprocess), **params
            )
        )

    return continual_dl, eval_dl, test_dl
