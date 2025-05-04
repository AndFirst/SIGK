import os
from abc import ABC, abstractmethod
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.classifier import Classifier
from src.dataset import DatasetWrapper
from src.results import Result, Results
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy, f1_score, precision, recall


class Experiment(ABC):
    def __init__(self, dataset_name: str, config: dict):
        self._dataset_name = dataset_name
        self._config = config
        self._real_dataset: DatasetWrapper = self._load_dataset(
            self._config["real_data_dir"], dataset_name
        )
        self._fake_dataset: DatasetWrapper = self._load_dataset(
            self._config["fake_data_dir"], dataset_name
        )
        self._classifier = Classifier(num_labels=self._real_dataset.get_num_labels())
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._results = Results()

    def run(self):
        self._train_classifier()
        self._test_classifier()
        self._save_plots()
        self._save_results()
        self._print_summary()

    def _print_summary(self) -> None:
        print(f"Experiment Summary: {self.NAME}")
        print(f"Dataset: {self._dataset_name}")
        print("Test Results:")
        for field_name, field_value in asdict(self._results["test"][0]).items():
            print(f"    {field_name}: {field_value}")

    def _save_results(self) -> None:
        os.makedirs(self._config["results_dir"], exist_ok=True)
        results_path = os.path.join(
            self._config["results_dir"],
            f"{self.__class__.__name__}_{self._dataset_name}.json",
        )
        self._results.save(results_path)

    def _test_classifier(self) -> None:
        test_loader = self._get_test_dataset()
        self._results.update("test", self._evaluate_classifier(test_loader))

    def _train_classifier(self) -> None:
        self._classifier.to(self._device)
        self._classifier.train()
        train_loader = self._get_train_dataset()
        val_loader = self._get_val_dataset()
        optimizer = torch.optim.Adam(
            self._classifier.parameters(), lr=self._config["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self._config["epochs"] + 1):
            print(f"Epoch {epoch}/{self._config['epochs']}")
            self._train_epoch(train_loader, optimizer, criterion)
            train_result = self._evaluate_classifier(train_loader)
            val_result = self._evaluate_classifier(val_loader)

            self._results.update("train", train_result)
            self._results.update("valid", val_result)

    def _evaluate_classifier(self, dataloader: DataLoader) -> Result:
        self._classifier.eval()
        criterion = nn.CrossEntropyLoss()
        num_classes = self._real_dataset.get_num_labels()

        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self._device), target.squeeze().to(self._device)
                output = self._classifier(data)
                total_loss += criterion(output, target).item()
                _, preds = torch.max(output, 1)
                all_preds.append(preds)
                all_targets.append(target)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return Result(
            loss=total_loss / len(dataloader),
            accuracy=accuracy(
                all_preds,
                all_targets,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ).item(),
            f1_score=f1_score(
                all_preds,
                all_targets,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ).item(),
            precision=precision(
                all_preds,
                all_targets,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ).item(),
            recall=recall(
                all_preds,
                all_targets,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ).item(),
        )

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        self._classifier.train()
        total_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(self._device), target.squeeze().to(self._device)
            optimizer.zero_grad()
            output = self._classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _save_plots(self) -> None:
        os.makedirs(self._config["results_dir"], exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{self.NAME} Learning Curves", fontsize=16)

        metrics = ["loss", "accuracy", "precision", "f1_score", "recall"]
        titles = ["Loss", "Accuracy", "Precision", "F1 Score", "Recall"]

        for ax, metric, title in zip(axes.flatten()[:5], metrics, titles):
            train_values = [
                getattr(result, metric) for result in self._results["train"]
            ]
            val_values = [getattr(result, metric) for result in self._results["valid"]]
            ax.plot(train_values, label=f"Train {title}")
            ax.plot(val_values, label=f"Validation {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(f"{title}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plot_path = os.path.join(
            self._config["results_dir"],
            f"{self.__class__.__name__}_{self._dataset_name}_learning_curves.png",
        )
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    def _load_dataset(self, dir_name: str, dataset_name: str) -> DatasetWrapper:
        path = os.path.join(dir_name, dataset_name)
        return DatasetWrapper(
            dir_name=path,
            config=self._config,
        )

    @abstractmethod
    def _get_train_dataset(self) -> DataLoader:
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def _get_val_dataset(self) -> DataLoader:
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def _get_test_dataset(self) -> DataLoader:
        raise NotImplementedError("This method should be overridden by subclasses")


class OriginalTrainFakeTest(Experiment):
    NAME = "Classifier trained on original data, tested on generated data"

    def _get_train_dataset(self) -> DataLoader:
        return self._real_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["train"]

    def _get_val_dataset(self) -> DataLoader:
        return self._real_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["val"]

    def _get_test_dataset(self) -> DataLoader:
        return self._fake_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["test"]


class FakeTrainOriginalTest(Experiment):
    NAME = "Classifier trained on generated data, tested on original data"

    def _get_train_dataset(self) -> DataLoader:
        return self._fake_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["train"]

    def _get_val_dataset(self) -> DataLoader:
        return self._fake_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["val"]

    def _get_test_dataset(self) -> DataLoader:
        return self._real_dataset.get_dataloaders(
            batch_size=self._config["batch_size"]
        )["test"]


class MixTrainMixTest(Experiment):
    NAME = "Classifier trained and tested on mixed original and generated data"

    def _get_train_dataset(self) -> DataLoader:
        return self._create_mixed_dataloader("train")

    def _get_val_dataset(self) -> DataLoader:
        return self._create_mixed_dataloader("val")

    def _get_test_dataset(self) -> DataLoader:
        return self._create_mixed_dataloader("test")

    def _create_mixed_dataloader(self, split: str) -> DataLoader:
        real_dataset = self._real_dataset.get_dataloaders(batch_size=1)[split].dataset
        fake_dataset = self._fake_dataset.get_dataloaders(batch_size=1)[split].dataset

        mixed_data = []
        mixed_labels = []

        num_labels = self._real_dataset.get_num_labels()

        for label in range(num_labels):
            real_indices = [
                i for i in range(len(real_dataset)) if real_dataset[i][1] == label
            ]
            fake_indices = [
                i for i in range(len(fake_dataset)) if fake_dataset[i][1] == label
            ]

            assert len(real_indices) == len(fake_indices), (
                "Datasets must have equal sizes per class"
            )

            real_selected = [real_indices[i] for i in range(1, len(real_indices), 2)]

            fake_selected = [fake_indices[i] for i in range(0, len(fake_indices), 2)]

            for idx in real_selected:
                data, lbl = real_dataset[idx]
                mixed_data.append(data)
                mixed_labels.append(lbl)
            for idx in fake_selected:
                data, lbl = fake_dataset[idx]
                mixed_data.append(data)
                mixed_labels.append(lbl)

        mixed_data_tensor = (
            torch.stack(mixed_data)
            if isinstance(mixed_data[0], torch.Tensor)
            else mixed_data
        )
        mixed_labels_tensor = torch.tensor(mixed_labels)

        mixed_dataset = TensorDataset(mixed_data_tensor, mixed_labels_tensor)

        return DataLoader(
            mixed_dataset,
            batch_size=self._config["batch_size"],
            shuffle=(split == "train"),
            num_workers=self._config.get("num_workers", 4),
        )
