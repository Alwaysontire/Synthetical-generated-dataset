import json, torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer
from model import TextCNN

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_PATH = "app_demo/model.pt"
LABELS_PATH = "app_demo/label_map.json"

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    id2label = json.load(file)

model = TextCNN(num_classes=len(id2label))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def classify(text: str):
    if not text.strip():
        return "Введите фразу", {}
    
    text = "query:" + text 
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=-1)[0]

        top_probs, top_ids = torch.topk(probs, k=5)

        result = {
            id2label[str(idx.item())]: float(prob.item())
            for prob, idx in zip(top_probs, top_ids)
        }

        best_label = id2label[str(top_ids[0].item())]
        confidence = top_probs[0].item()

        ans = f"{best_label} ({confidence:.2f})"

        return ans, result
    

examples = [
    "Включи подогрев сиденья",
    "Сделай музыку тише",
    "Открой окно водителя",
    "Построй маршрут до дома",
    "Turn on the air conditioning",
    "Call mom",
    "Switch to sport mode",
]
    
demo = gr.Interface(
    fn=classify, 
    inputs=gr.Textbox(
        label="Фраза водителя",
        placeholder="Например: включи кондиционер на 22 градуса",
        lines=2
    ),
    outputs=[
        gr.Textbox(label="Предсказанный интент"),
        gr.Label(label="Топ 5 вероятностей")
    ],
    examples=examples,
    title="Классификация интентов водителя",
    description=(
        "Демонстрация работы модели классификации команд автомобильного голосового ассистента."
        "Введите фразу на русском или английском языке, модель определит соответствующий интент."
    ),
    theme="soft"
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
