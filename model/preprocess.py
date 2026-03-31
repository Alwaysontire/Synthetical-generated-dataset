import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch, os, json

def drop_nan(data : pd.DataFrame):
    data.dropna(subset=["phrase", "intent"], inplace=True)
    return data


def class_balance(data):
    pass

def LabelEnconding(data):
    le = LabelEncoder()
    target = data["intent"]
    target_transformed = le.fit_transform(target)
    data["intent"] = target_transformed
    return data, le


def tokenize_save(tokenizer, X, y, path, max_length=64):
    texts = ("query: " + X["phrase"].astype(str)).tolist()
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    torch.save({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(y.tolist(), dtype=torch.long)
    }, path)

def main():
    data = pd.read_csv("samples/multilingual_data.csv")
    data = drop_nan(data)
    data_enc, le = LabelEnconding(data)
    y = data_enc["intent"]
    X = data_enc[["phrase", "parameters"]]
    X_train, X_1, y_train, y_1 = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_1, y_1, train_size=0.5, random_state=42, shuffle=True, stratify=y_1)
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")


    tokenize_save(tokenizer, X_train, y_train, "data/large_train.pt")
    tokenize_save(tokenizer, X_val, y_val, "data/large_val.pt")
    tokenize_save(tokenizer, X_test, y_test, "data/large_test.pt")



if __name__ == "__main__":
    main()
