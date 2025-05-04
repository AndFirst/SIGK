import glob
import json
import os
from typing import Type

import matplotlib.pyplot as plt
from medmnist.dataset import MedMNIST2D
from PIL import Image


def save_medmnist_to_dirs(dataset_class: Type[MedMNIST2D], output_dir):
    dataset_dir = os.path.join(output_dir, dataset_class.__name__)
    os.makedirs(dataset_dir, exist_ok=True)
    train_dataset = dataset_class(split="train", download=True, size=64)
    val_dataset = dataset_class(split="val", download=True, size=64)
    test_dataset = dataset_class(split="test", download=True, size=64)
    splits = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    for split_name, dataset in splits.items():
        split_dir = os.path.join(dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        images, labels = dataset.imgs, dataset.labels
        class_counters = {}
        for img, label in zip(images, labels):
            label = int(label[0]) if label.ndim > 0 else int(label)
            class_dir = os.path.join(split_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            if label not in class_counters:
                class_counters[label] = 0
            img_filename = f"{class_counters[label]:04d}.png"
            img_path = os.path.join(class_dir, img_filename)
            img_pil = Image.fromarray(img)
            img_pil.save(img_path)
            class_counters[label] += 1
    print(f"Dane z {dataset_class.__name__} zostały zapisane w {dataset_dir}")


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_metrics(experiment_name, datasets_data, dataset_colors):
    metrics = ["loss", "accuracy", "precision", "f1_score", "recall"]
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Experiment: {experiment_name}", fontsize=16)
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        max_epochs = 0
        for dataset_name, data in datasets_data.items():
            epochs = range(1, len(data["train"][metric]) + 1)
            max_epochs = max(max_epochs, len(epochs))
            color = dataset_colors[dataset_name]
            plt.plot(
                epochs,
                data["train"][metric],
                label=f"{dataset_name} Train",
                color=color,
                linestyle="-",
                marker="o",
                alpha=1.0,
            )
            plt.plot(
                epochs,
                data["valid"][metric],
                label=f"{dataset_name} Valid",
                color=color,
                linestyle="--",
                marker="s",
                alpha=0.6,
            )
            test_value = data["test"][metric][0]
            plt.axhline(
                y=test_value,
                label=f"{dataset_name} Test",
                color=color,
                linestyle=":",
                alpha=0.8,
            )
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison")
        plt.legend()
        plt.grid(True)
        plt.xlim(1, max_epochs)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_experiment(experiment_name, results_dir="results"):
    json_files = glob.glob(os.path.join(results_dir, f"{experiment_name}_*.json"))
    if not json_files:
        print(f"No files found for experiment: {experiment_name}")
        return
    datasets_data = {}
    dataset_names = set()
    for file_path in json_files:
        filename = os.path.basename(file_path).replace(".json", "")
        _, dataset_name = filename.split("_", 1)
        data = load_data(file_path)
        datasets_data[dataset_name] = {
            "train": data["train"],
            "valid": data["valid"],
            "test": data["test"],
        }
        dataset_names.add(dataset_name)
    colors = ["blue", "red", "green", "purple"]
    dataset_colors = {
        name: colors[i % len(colors)] for i, name in enumerate(sorted(dataset_names))
    }
    plot_metrics(experiment_name, datasets_data, dataset_colors)
