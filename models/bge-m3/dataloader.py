from torch.utils.data import Dataset, DataLoader
import torch, os

class PhraseDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = torch.load(path, weights_only=True)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index]
        }
    

def build_dataloader(processed_dir, batch_size=256, num_workers=0):
    train_ds = PhraseDataset(os.path.join(processed_dir, "train.pt"))
    val_ds = PhraseDataset(os.path.join(processed_dir, "val.pt"))
    test_ds = PhraseDataset(os.path.join(processed_dir, "test.pt"))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl


