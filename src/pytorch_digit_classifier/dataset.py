import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_digit_classifier.data import DatasetSplit, PreparedDataset


class MNISTDataset(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        self.images = torch.as_tensor(split.images, dtype=torch.float32)
        self.labels = torch.as_tensor(split.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def create_dataloaders(
    dataset: PreparedDataset,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        MNISTDataset(dataset.train),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        MNISTDataset(dataset.validation),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MNISTDataset(dataset.test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader
