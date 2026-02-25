from dataloader import build_dataloader
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class TextCNN(nn.Module):
    def __init__(self, num_classes=90, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), embedding_dim=256, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        
        hidden = self.backbone.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, num_filters, k) for k in kernel_sizes
        ])
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(kernel_sizes), embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        x = hidden.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = c.max(dim=2).values
            pooled.append(c)
        x = torch.cat(pooled, dim=1)
        embedding = self.projection(x)
        return embedding
    

    def forward(self, input_ids, attention_mask):
        embedding = self.encode(input_ids, attention_mask)
        logits = self.classifier(embedding)
        return logits, embedding





def main():
    train, val, test = build_dataloader("data/processed")
    
    model = TextCNN()
    model.eval()

    batch = next(iter(train))
    with torch.no_grad():
        logits, embedding = model(batch["input_ids"], batch["attention_mask"])
    print("log shape: ", logits.shape)
    print("emd shape: ", embedding.shape)

if __name__ == "__main__":
    main()