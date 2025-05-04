import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.dataset import DatasetWrapper
from src.generator import ConditionalDiffusionModel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from tqdm import tqdm


def train_on_dataset(dataset: DatasetWrapper, config: Dict[str, Any]) -> None:
    epochs = config["epochs"]
    checkpoint_dir = config["checkpoint_dir"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    num_labels = dataset.get_num_labels()
    model = ConditionalDiffusionModel(num_labels=num_labels, **config)

    dataloaders = dataset.get_dataloaders(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["patience"],
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_train_samples = 0

        with tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.squeeze().to(device)
                batch_size = x.shape[0]
                num_train_samples += batch_size

                optimizer.zero_grad()
                t = torch.randint(
                    0, model.num_timesteps, (batch_size,), device=device
                ).long()
                x_noisy, noise = model.forward_diffusion(x, t, device)
                predicted_noise = model(x_noisy, t, y)
                loss = F.mse_loss(predicted_noise, noise, reduction="sum") / (
                    batch_size * x.shape[1] * x.shape[2] * x.shape[3]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_size
                pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss / num_train_samples:.6f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                    }
                )

        train_loss /= num_train_samples

        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for x, y in dataloaders["val"]:
                x = x.to(device)
                y = y.squeeze().to(device)
                batch_size = x.shape[0]
                num_val_samples += batch_size

                t = torch.randint(
                    0, model.num_timesteps, (batch_size,), device=device
                ).long()
                x_noisy, noise = model.forward_diffusion(x, t, device)
                predicted_noise = model(x_noisy, t, y)
                loss = F.mse_loss(predicted_noise, noise, reduction="sum") / (
                    batch_size * x.shape[1] * x.shape[2] * x.shape[3]
                )
                val_loss += loss.item() * batch_size

        val_loss /= num_val_samples
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, "
            f"Val Loss={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_{dataset.name}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model at epoch {epoch + 1} with Val Loss={val_loss:.6f}")

    final_checkpoint_path = os.path.join(
        checkpoint_dir, f"last_model_{dataset.name}.pt"
    )
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Saved final model at {final_checkpoint_path}")


def visualize_best_model(dataset: Any, config: Dict[str, Any]) -> None:
    checkpoint_dir = config["checkpoint_dir"]
    num_labels = dataset.get_num_labels()
    model = ConditionalDiffusionModel(num_labels=num_labels, **config)

    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{dataset.name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloaders = dataset.get_dataloaders(batch_size=config.get("batch_size", 32))
    val_loader = dataloaders["val"]

    n_classes = num_labels
    num_samples_per_class = 3

    real_images = {i: [] for i in range(n_classes)}
    with torch.no_grad():
        for x, y in val_loader:
            y = y.squeeze()
            for i in range(len(y)):
                class_idx = y[i].item()
                if len(real_images[class_idx]) < num_samples_per_class:
                    real_images[class_idx].append(x[i])
                if all(
                    len(real_images[c]) >= num_samples_per_class
                    for c in range(n_classes)
                ):
                    break
            if all(
                len(real_images[c]) >= num_samples_per_class for c in range(n_classes)
            ):
                break

    fake_images = []
    with torch.no_grad():
        for class_idx in range(n_classes):
            samples = model.sample(
                label=class_idx,
                num_samples=num_samples_per_class,
                device=device,
                batch_size=num_samples_per_class,
            ).cpu()
            fake_images.append((samples + 1) / 2)

    fig, axs = plt.subplots(
        n_classes,
        num_samples_per_class * 2,
        figsize=(num_samples_per_class * 4, n_classes * 2),
    )
    fig.suptitle("Real vs Generated Samples per Class", fontsize=16)

    for class_idx in range(n_classes):
        for i in range(num_samples_per_class):
            ax = axs[class_idx, i] if n_classes > 1 else axs[i]
            img = real_images[class_idx][i].permute(1, 2, 0).squeeze()
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            ax.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(f"Class {class_idx}", fontsize=12)
            if class_idx == 0 and i == 0:
                ax.set_title("Real", fontsize=12)

        for i in range(num_samples_per_class):
            ax = (
                axs[class_idx, i + num_samples_per_class]
                if n_classes > 1
                else axs[i + num_samples_per_class]
            )
            img = fake_images[class_idx][i].permute(1, 2, 0).squeeze()
            ax.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
            ax.axis("off")
            if class_idx == 0 and i == 0:
                ax.set_title("Generated", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"real_vs_generated_{dataset.name}.png")


def generate_fake_data(dataset: Any, config: Dict[str, Any]) -> None:
    checkpoint_dir = config["checkpoint_dir"]
    output_dir = config["generated_data_dir"]
    dataset_name = dataset.name
    sample_sizes = config["sample_sizes"]
    batch_size = config.get("batch_size", 64)

    base_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)

    num_labels = dataset.get_num_labels()
    model = ConditionalDiffusionModel(num_labels=num_labels, **config)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{dataset_name}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for split in ["train", "val", "test"]:
        samples_per_class = sample_sizes[split]
        print(f"Generating split: {split} with {samples_per_class} samples/class")

        for class_idx in tqdm(range(num_labels), desc=f"{split} classes"):
            save_dir = os.path.join(base_output_dir, split, str(class_idx))
            os.makedirs(save_dir, exist_ok=True)

            num_batches = (samples_per_class + batch_size - 1) // batch_size
            img_counter = 0

            for _ in range(num_batches):
                current_batch_size = min(batch_size, samples_per_class - img_counter)
                images = model.sample(
                    label=class_idx,
                    num_samples=current_batch_size,
                    device=device,
                    batch_size=current_batch_size,
                )
                images = (images + 1) / 2
                images = images.clamp(0, 1)

                for i in range(images.size(0)):
                    save_path = os.path.join(save_dir, f"{img_counter:04d}.png")
                    save_image(images[i], save_path)
                    img_counter += 1
