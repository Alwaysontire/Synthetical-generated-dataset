import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from model import BertModel
from dataloader import build_dataloader
import torch.nn as nn
import tqdm
from config import processed_path
from transformers import get_cosine_schedule_with_warmup


def train_epoch(model, loader, optimizer, criterion, scheduler, DEVICE):
    model.train()

    mean_loss = []
    mean_acc = []

    for batch in loader:
        inputs_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(inputs_ids, attention_mask)
        acc = (torch.argmax(logits, dim=1) == label).float().mean()
        loss = criterion(logits, label)

        mean_loss.append(loss.item())
        mean_acc.append(acc.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return np.array(mean_loss).mean(), np.array(mean_acc).mean()

def evaluate_epoch(model, loader, criterion, DEVICE):
    model.eval()

    mean_loss = []
    mean_acc = []

    for batch in loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, label)
            acc = (torch.argmax(logits, dim=1) == label).float().mean()

            mean_loss.append(loss.item())
            mean_acc.append(acc.item())

    return np.array(mean_loss).mean(), np.array(mean_acc).mean()


def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    LR_BERT = 1e-5
    LR_HEAD = 4e-4

    model = BertModel().to(DEVICE)
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr" : LR_BERT},
                                   {"params": model.classification.parameters(), "lr": LR_HEAD}])
    
    train_loader, val_loader, test_loader = build_dataloader(processed_path("mmBERT"))
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.09)
    PATIENCE = 4
    best_val_acc = 0
    patience_counter = 0
    
    losses = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "current_lr": [],
        "epoch": []
    }

    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, DEVICE)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, DEVICE)

        losses["train_loss"].append(train_loss)
        losses["train_acc"].append(train_acc)
        losses["val_loss"].append(val_loss)
        losses["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[1]["lr"]

        losses["current_lr"].append(current_lr)
        losses["epoch"].append(epoch)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "data/best_models/mmBERT/model.pt")
        else:
            patience_counter += 1

        print(f"EPOCH {epoch + 1}: train_loss = {train_loss:.3f}, train_acc = {train_acc:.3f}, val_loss = {val_loss:.3f}, val_acc = {val_acc:.3f}, lr_head = {current_lr:.6f}")

        if patience_counter >= PATIENCE:
            break


if __name__ == "__main__":
    main()













    