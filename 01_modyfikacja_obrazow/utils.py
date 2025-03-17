import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def set_seed(seed_value):
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed_value=42)


class SNE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "sum_squared_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        diff = preds - target
        squared_diff = torch.sum(diff**2)
        self.sum_squared_diff += squared_diff

    def compute(self):
        return self.sum_squared_diff


class MetricContainer(dict):
    def __add__(self, other):
        result = MetricContainer()
        for key in self.keys():
            result[key] = self[key] + other[key]
        return result

    def __truediv__(self, number: float):
        result = MetricContainer()
        for key in self.keys():
            result[key] = self[key] / number
        return result

    def __str__(self):
        return ", ".join(f"{key}: {value:.4f}" for key, value in self.items())


def compute_metrics(image_1: torch.Tensor, image_2: torch.Tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_1 = image_1.to(device)
    image_2 = image_2.to(device)

    if len(image_1.shape) == 3:
        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)

    psnr_metric = PSNR().to(device)
    lpips_metric = LPIPS().to(device)
    ssim_metric = SSIM().to(device)
    sne_metric = SNE().to(device)

    psnr_value = psnr_metric(image_1, image_2)
    lpips_value = lpips_metric(image_1, image_2)
    ssim_value = ssim_metric(image_1, image_2)
    sne_value = sne_metric(image_1, image_2)
    return MetricContainer(
        {
            "PSNR": psnr_value.item(),
            "LPIPS": lpips_value.item(),
            "SSIM": ssim_value.item(),
            "SNE": sne_value.item(),
        }
    )


class ImageVisualizer:
    def __init__(self, column_names: list):
        self.n_columns = len(column_names)
        self.column_names = column_names
        self.images = []

    def add_images(self, image_list: list):
        if len(image_list) != self.n_columns:
            raise ValueError(
                f"Lista obrazów musi zawierać dokładnie {self.n_columns} elementów."
            )

        for i, image in enumerate(image_list):
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            elif len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                pass
            else:
                raise ValueError(
                    f"Nieobsługiwany format obrazu {i}. Oczekiwano [H, W] lub [H, W, C]."
                )

            image = np.clip(image, 0, 1)

            self.images.append(image)

    def show_images(self):
        n_images = len(self.images)
        n_rows = (n_images + self.n_columns - 1) // self.n_columns

        fig, axes = plt.subplots(
            n_rows, self.n_columns, figsize=(self.n_columns * 4, n_rows * 4)
        )

        if n_rows == 1:
            axes = (
                np.array([axes])
                if self.n_columns == 1
                else np.expand_dims(axes, axis=0)
            )

        for i in range(n_rows):
            for j in range(self.n_columns):
                idx = i * self.n_columns + j
                ax = axes[i, j]

                if idx < n_images:
                    ax.imshow(self.images[idx])
                else:
                    ax.imshow(np.zeros_like(self.images[0]))

                if i == 0:
                    ax.set_title(self.column_names[j])
                else:
                    ax.set_title("")

                ax.axis("off")

        plt.tight_layout()
        plt.show()
