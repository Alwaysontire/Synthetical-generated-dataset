import torch
import numpy as np
import json
from sklearn.metrics import classification_report, precision_recall_fscore_support
from model import BertModel
from dataloader import build_dataloader
from sklearn.metrics import confusion_matrix


def evaluate(model, test_dl, DEVICE):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open("data/processed/qwen2/label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]

    test_loader = torch.load("data/processed/qwen2/test.pt")
    # _, _, test_loader = build_dataloader("data/")

    model = BertModel().to(DEVICE)
    model.load_state_dict(torch.load("data/best_models/qwen2/model.pt", map_location=DEVICE))

    labels, preds = evaluate(model, test_loader, DEVICE)

    acc = (labels == preds).mean()
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro    P/R/F1: {p_macro:.3f} / {r_macro:.3f} / {f1_macro:.3f}")
    print(f"Weighted P/R/F1: {p_weighted:.3f} / {r_weighted:.3f} / {f1_weighted:.3f}")
    print(classification_report(labels, preds, target_names=class_names, digits=3))
    print(confusion_matrix(labels, preds, labels=None))


if __name__ == "__main__":
    main()
