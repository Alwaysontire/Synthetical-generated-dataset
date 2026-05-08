from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class TextCNN(nn.Module):
    def __init__(self, num_classes=82, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), embedding_dim=256, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
        hidden = self.backbone.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, num_filters, k) for k in kernel_sizes
        ])

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(kernel_sizes) + hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def masked_mean_pool(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, 0.0)
        
        return hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        x = hidden.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.gelu(conv(x))
            c = c.max(dim=2).values
            pooled.append(c)

        cnn_feat = torch.cat(pooled, dim=1)
        mean_feat = self.masked_mean_pool(hidden, attention_mask)

        embedding = self.projection(torch.cat([cnn_feat, mean_feat], dim=1))

        return embedding
    

    def forward(self, input_ids, attention_mask):
        embedding = self.encode(input_ids, attention_mask)
        logits = self.classifier(embedding)
        return logits, embedding


