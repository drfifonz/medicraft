from torch.utils.data import Dataset


class EyeFundusDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        pass

    def __len__(self) -> int:
        pass
