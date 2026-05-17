import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import tqdm

from transformers import get_cosine_schedule_with_warmup

from model import BertModel
from dataloader import build_dataloader
from config import processed_path


def hierarchy_loss(
    domain_logits,
    intent_logits_by_domain,
    domain_labels,
    local_intent_labels,
    criterion,
    domain_loss_weight=0.3
):
    domain_loss = criterion(domain_logits, domain_labels)

    intent_loss = 0.0
    total_count = 0

    for domain_id, local_logits in intent_logits_by_domain.items():
        domain_id_int = int(domain_id)
        mask = domain_labels == domain_id_int

        if mask.any():
            logits_d = local_logits[mask]
            targets_d = local_intent_labels[mask]

            loss_d = criterion(logits_d, targets_d)

            count_d = mask.sum().item()
            intent_loss = intent_loss + loss_d * count_d
            total_count += count_d

    intent_loss = intent_loss / max(total_count, 1)

    loss = intent_loss + domain_loss_weight * domain_loss

    return loss, intent_loss, domain_loss


def hierarchy_predict(
    domain_logits,
    intent_logits_by_domain,
    domain_to_intents,
    device
):
    pred_domains = torch.argmax(domain_logits, dim=1)

    batch_size = domain_logits.size(0)

    global_intent_preds = torch.empty(
        batch_size,
        dtype=torch.long,
        device=device
    )

    for domain_id, local_logits in intent_logits_by_domain.items():
        domain_id_int = int(domain_id)
        mask = pred_domains == domain_id_int

        if mask.any():
            local_preds = torch.argmax(local_logits[mask], dim=1)

            global_ids = torch.tensor(
                domain_to_intents[str(domain_id_int)],
                dtype=torch.long,
                device=device
            )

            global_intent_preds[mask] = global_ids[local_preds]

    return global_intent_preds, pred_domains


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scheduler,
    domain_to_intents,
    device,
    domain_loss_weight=0.3
):
    model.train()

    mean_loss = []
    mean_intent_loss = []
    mean_domain_loss = []
    mean_intent_acc = []
    mean_domain_acc = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        intent_label = batch["intent_label"].to(device)
        domain_label = batch["domain_label"].to(device)
        local_intent_label = batch["local_intent_label"].to(device)

        optimizer.zero_grad()

        domain_logits, intent_logits_by_domain = model(
            input_ids,
            attention_mask
        )

        loss, intent_loss, domain_loss = hierarchy_loss(
            domain_logits=domain_logits,
            intent_logits_by_domain=intent_logits_by_domain,
            domain_labels=domain_label,
            local_intent_labels=local_intent_label,
            criterion=criterion,
            domain_loss_weight=domain_loss_weight
        )

        global_intent_preds, pred_domains = hierarchy_predict(
            domain_logits=domain_logits,
            intent_logits_by_domain=intent_logits_by_domain,
            domain_to_intents=domain_to_intents,
            device=device
        )

        intent_acc = (global_intent_preds == intent_label).float().mean()
        domain_acc = (pred_domains == domain_label).float().mean()

        mean_loss.append(loss.item())
        mean_intent_loss.append(intent_loss.item())
        mean_domain_loss.append(domain_loss.item())
        mean_intent_acc.append(intent_acc.item())
        mean_domain_acc.append(domain_acc.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

    return {
        "loss": np.mean(mean_loss),
        "intent_loss": np.mean(mean_intent_loss),
        "domain_loss": np.mean(mean_domain_loss),
        "intent_acc": np.mean(mean_intent_acc),
        "domain_acc": np.mean(mean_domain_acc),
    }


def evaluate_epoch(
    model,
    loader,
    criterion,
    domain_to_intents,
    device,
    domain_loss_weight=0.3
):
    model.eval()

    mean_loss = []
    mean_intent_loss = []
    mean_domain_loss = []
    mean_intent_acc = []
    mean_domain_acc = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            intent_label = batch["intent_label"].to(device)
            domain_label = batch["domain_label"].to(device)
            local_intent_label = batch["local_intent_label"].to(device)

            domain_logits, intent_logits_by_domain = model(
                input_ids,
                attention_mask
            )

            loss, intent_loss, domain_loss = hierarchy_loss(
                domain_logits=domain_logits,
                intent_logits_by_domain=intent_logits_by_domain,
                domain_labels=domain_label,
                local_intent_labels=local_intent_label,
                criterion=criterion,
                domain_loss_weight=domain_loss_weight
            )

            global_intent_preds, pred_domains = hierarchy_predict(
                domain_logits=domain_logits,
                intent_logits_by_domain=intent_logits_by_domain,
                domain_to_intents=domain_to_intents,
                device=device
            )

            intent_acc = (global_intent_preds == intent_label).float().mean()
            domain_acc = (pred_domains == domain_label).float().mean()

            mean_loss.append(loss.item())
            mean_intent_loss.append(intent_loss.item())
            mean_domain_loss.append(domain_loss.item())
            mean_intent_acc.append(intent_acc.item())
            mean_domain_acc.append(domain_acc.item())

    return {
        "loss": np.mean(mean_loss),
        "intent_loss": np.mean(mean_intent_loss),
        "domain_loss": np.mean(mean_domain_loss),
        "intent_acc": np.mean(mean_intent_acc),
        "domain_acc": np.mean(mean_domain_acc),
    }


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 10
    LR_BERT = 1e-5
    LR_HEAD = 4e-4
    WARMUP_RATIO = 0.06
    DOMAIN_LOSS_WEIGHT = 0.3
    PATIENCE = 3

    PROCESSED_DIR = processed_path("bge-m3")

    with open(PROCESSED_DIR / "domain_to_intents.json", "r", encoding="utf-8") as file:
        domain_to_intents = json.load(file)

    num_domains = len(domain_to_intents)

    model = BertModel(
        num_domains=num_domains,
        domain_to_intents=domain_to_intents
    ).to(DEVICE)

    head_params = (
        list(model.projection.parameters())
        + list(model.domain_head.parameters())
        + list(model.intent_heads.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": LR_BERT
            },
            {
                "params": head_params,
                "lr": LR_HEAD
            }
        ],
        weight_decay=0.01
    )

    train_loader, val_loader, test_loader = build_dataloader(PROCESSED_DIR)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_intent_acc = 0.0
    patience_counter = 0

    Path("data/best_models/bge-m3").mkdir(parents=True, exist_ok=True)

    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            domain_to_intents=domain_to_intents,
            device=DEVICE,
            domain_loss_weight=DOMAIN_LOSS_WEIGHT
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            domain_to_intents=domain_to_intents,
            device=DEVICE,
            domain_loss_weight=DOMAIN_LOSS_WEIGHT
        )

        current_lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"EPOCH {epoch + 1}: "
            f"train_loss = {train_metrics['loss']:.3f}, "
            f"train_intent_acc = {train_metrics['intent_acc']:.3f}, "
            f"train_domain_acc = {train_metrics['domain_acc']:.3f}, "
            f"val_loss = {val_metrics['loss']:.3f}, "
            f"val_intent_acc = {val_metrics['intent_acc']:.3f}, "
            f"val_domain_acc = {val_metrics['domain_acc']:.3f}, "
            f"lr_head = {current_lr_head:.6f}"
        )

        if val_metrics["intent_acc"] > best_val_intent_acc:
            best_val_intent_acc = val_metrics["intent_acc"]
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "data/best_models/bge-m3/model_hierarchical.pt"
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best val_intent_acc = {best_val_intent_acc:.4f}"
            )
            break


if __name__ == "__main__":
    main()