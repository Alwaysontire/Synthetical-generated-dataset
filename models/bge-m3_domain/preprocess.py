import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch, json
from config import RAW_DATA, processed_path

MODEL_DIR = processed_path("bge-m3")

def drop_nan(data : pd.DataFrame):
    data = data.copy()
    data.dropna(subset=["phrase", "intent", "domain"], inplace=True)
    return data
    

def LabelEnconding(data):
    data = data.copy()
    intent_encoder = LabelEncoder()
    domain_encoder = LabelEncoder()



    data["intent_label"] = intent_encoder.fit_transform(data["intent"])
    data["domain_label"] = domain_encoder.fit_transform(data["domain"])

    return data, intent_encoder, domain_encoder

def hierarchy_maps(data, intent_encoder, domain_encoder):
    domain_to_intents = {}
    local_maps = {}

    for domain_name in sort(data["domain"].unique()):
        domain_id = int(domain_encoder.transform([domain_name])[0])

        intents_in_domain = sorted(
            data.loc[data["domain"] == domain_name, "intents"].unique()
        )

        global_intent_ids = [
            int(intent_encoder.transform([intent_name])[0])
            for intent_name in intents_in_domain
        ]
        domain_to_intents[str(domain_id)] = global_intent_ids

        local_maps[domain_name] = {
            intent_name: local_id
            for local_id, intent_name in enumerate(intents_in_domain)
        }

        data["local_intent_label"] = data.apply(
            lambda row: local_maps[row["domain"]][row["intent"]], axis=1
        )
    return data, domain_to_intents

def save_maps(intent_encoder, domain_encoder, domain_to_intents, path: Path):
    labels_map_intent = {
        int(i): str(cls)
        for i, cls in enumerate(intent_encoder.classes_)
    }

    label_map_domain = {
        int(i): str(cls)
        for i, cls in enumerate(domain_encoder.classes_)
    }
    with open(path / "label_map_intent.json", "w", encoding="utf-8") as file:
        json.dump(label_map_intent, file, ensure_ascii=False, indent=2)

    with open(path / "label_map_domain.json", "w", encoding="utf-8") as file:
        json.dump(label_map_domain, file, ensure_ascii=False, indent=2)

    with open(path / "domain_to_intents.json", "w", encoding="utf-8") as file:
        json.dump(domain_to_intents, file, ensure_ascii=False, indent=2)


def tokenize_save(tokenizer, data, path, max_length=32):

    enc = tokenizer(
        X["phrase"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    torch.save({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],

        "labels": torch.tensor(data["intent_label"].tolist(), dtype=torch.long),
        "intent_labels": torch.tensort(data["intent_label"].tolist(), dtype=torch.long),

        "domain_labels": torch.tensor(data["domain_label"].tolist(), dtype=torch.long),
        "local_intent_labels": torch.tensor(data["local_intent_label"].tolist(), dtype=torch.long)
        
    }, path)

def main():

    data = pd.read_csv(RAW_DATA)
    data = drop_nan(data)
    data_enc, intent_encoder, domain_encoder = LabelEnconding(data)
    data, domain_to_intents = hierarchy_maps(data, intent_encoder, domain_encoder)

    train_df, temp_df = train_test_split(data_enc, train_size=0.8, random_state=42, shuffle=True, stratify=data_enc["intent_label"])
    val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42, shuffle=True, stratify=temp_df["intent_label"])

    

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_maps(
        intent_encoder, 
        domain_encoder,
        domain_to_intents,
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    tokenize_save(tokenizer, X_train, y_train, MODEL_DIR / "train.pt")
    tokenize_save(tokenizer, X_val, y_val, MODEL_DIR / "val.pt")
    tokenize_save(tokenizer, X_test, y_test, MODEL_DIR / "test.pt")


if __name__ == "__main__":
    main()
