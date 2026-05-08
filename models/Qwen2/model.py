import torch.nn as nn
from transformers import AutoModel
import torch
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig


class BertModel(nn.Module):
    def __init__(self, num_classes=82, embedding_dim=256):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            torch_dtype=torch.bfloat16
        )
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        self.backbone = get_peft_model(self.backbone, lora_config)
        hidden_size = self.backbone.config.hidden_size 

        self.classification = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.float()

        last_token_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0)), last_token_idx]

        logits = self.classification(pooled)
        return logits
