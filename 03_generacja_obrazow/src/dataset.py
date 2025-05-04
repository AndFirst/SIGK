import os
import random

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DatasetWrapper:
    def __init__(self, dir_name: str, config: dict):
        self._dir_name = str(dir_name)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._train_dataset = self._load_dataset("train")
        self._val_dataset = self._load_dataset("val")
        self._test_dataset = self._load_dataset("test")
        if config.get("sample_sizes"):
            self._train_dataset = self._get_sampled_data(
                "train", config["sample_sizes"]["train"]
            )
            self._val_dataset = self._get_sampled_data(
                "val", config["sample_sizes"]["val"]
            )
            self._test_dataset = self._get_sampled_data(
                "test", config["sample_sizes"]["test"]
            )

    @property
    def name(self) -> str:
        return os.path.basename(self._dir_name)

    def get_num_labels(self) -> int:
        return len(os.listdir(os.path.join(self._dir_name, "train")))

    def get_dataloaders(self, batch_size: int) -> dict[str, DataLoader]:
        return {
            "train": DataLoader(
                self._train_dataset, batch_size=batch_size, num_workers=4, shuffle=True
            ),
            "val": DataLoader(
                self._val_dataset, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            "test": DataLoader(
                self._test_dataset, batch_size=batch_size, num_workers=4, shuffle=False
            ),
        }

    def _load_dataset(self, split: str) -> ImageFolder:
        return ImageFolder(
            root=os.path.join(self._dir_name, split),
            transform=self._transform,
        )

    def _get_sampled_data(self, split: str, sample_size: int) -> Dataset:
        if sample_size is None:
            return self._load_dataset(split)
        dataset = self._load_dataset(split)
        class_to_indices = {}
        for idx, (_, class_idx) in enumerate(dataset.samples):
            class_idx = int(class_idx)
            if class_idx not in class_to_indices:
                class_to_indices[class_idx] = []
            class_to_indices[class_idx].append(idx)
        new_indices = []
        for class_idx in class_to_indices:
            indices = class_to_indices[class_idx]
            current_size = len(indices)
            if current_size >= sample_size:
                selected_indices = random.sample(indices, sample_size)
            else:
                selected_indices = indices.copy()
                additional_indices = random.choices(
                    indices, k=sample_size - current_size
                )
                selected_indices.extend(additional_indices)
            new_indices.extend(selected_indices)
        return Subset(dataset, new_indices)

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
