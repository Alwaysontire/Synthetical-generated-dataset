import torch.nn as nn
from transformers import AutoModel
import torch


class BertModel(nn.Module):
    def __init__(self, num_classes=64, embedding_dim=256):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("jhu-clsp/mmBERT-base")
        hidden_size = self.backbone.config.hidden_size

        self.classification = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out_bert = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out_bert.last_hidden_state
        cls_token = hidden[:, 0, :] 

        mask = attention_mask.unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1)

        max_hidden = hidden.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        max_pool = max_hidden.max(dim=1)[0]

        pooled = torch.cat([cls_token, mean, max_pool], dim=-1)

        logits = self.classification(pooled)
        return logits



