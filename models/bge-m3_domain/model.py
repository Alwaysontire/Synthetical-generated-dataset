import torch.nn as nn
from transformers import AutoModel
import torch


class BertModel(nn.Module):
    def __init__(self, num_domains, domain_to_intents, embedding_dim=256):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("BAAI/bge-m3")
        hidden_size = self.backbone.config.hidden_size
        pooled_size = hidden_size * 3


        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Dropout(p=0.2),
            nn.Linear(pooled_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )

        self.domain_head = nn.Linear(embedding_dim, num_domains)
        self.intent_heads = nn.ModuleDict()

        for domain_id, intent_ids in domain_to_intents.items():
            self.intent_heads[str(domain_id)] = nn.Linear(embedding_dim, len(intent_ids))
        
    
    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden = out.last_hidden_state
        cls_token = hidden[:, 0, :]

        mask = attention_mask.unsqueeze(dim=-1).float()
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        max_hidden = hidden.masked_fill(attention_mask.unsqueeze(dim=-1) == 0, -1e4)
        max_pool = max_hidden.max(dim=1)[0]

        pooled = torch.cat([cls_token, mean_pool, max_pool], dim=-1)
        features = self.projection(pooled)

        return features



    def forward(self, input_ids, attention_mask):
        features = self.encode(input_ids, attention_mask)
        domain_logits = self.domain_head(features)
        intent_logits_by_domain = {}

        for domain_id, head in self.intent_heads.items():
            intent_logits_by_domain[domain_id] = head(features)

        return domain_logits, intent_logits_by_domain



