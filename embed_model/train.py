import torch
import numpy as np
from model import TextCNN
from dataloader import build_dataloader
from sklearn.utils.class_weight import compute_class_weight


def train_epoch(model, loader, optimizer, criterion, DEVICE):
    model.train()
    mean_loss = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)


        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, label)
        mean_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    return np.array(mean_loss).mean()

def eval_epoch(model, loader, criterion, DEVICE):
    model.eval()
    mean_loss = []
    mean_acc = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, label)
            acc = (logits.argmax(dim=1) == label).float().mean()
            mean_loss.append(loss.item())
            mean_acc.append(acc.item())
    return np.array(mean_loss).mean(), np.array(mean_acc).mean()



def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    EPOCHS = 20
    LR = 1e-3

    train_dl, val_dl, _ = build_dataloader("data/processed")
    train_labels = train_dl.dataset.labels.numpy()
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    weights = np.clip(weights, 1.0, weights.mean() * 3)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS)
        
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_dl, criterion, DEVICE)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "data/best_model.pt")
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, lr={current_lr:.7f}")



if __name__ == "__main__":
    main()