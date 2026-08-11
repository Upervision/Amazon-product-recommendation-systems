"""
User-Item Transformer Recommender (UIT-Rec)
Complete single-file implementation for Amazon Electronics rating prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# -------------------------------------------------
# 1. Dataset
# -------------------------------------------------
class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings  = torch.tensor(ratings,  dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

# -------------------------------------------------
# 2. Model
# -------------------------------------------------
class UserItemTransformer(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.type_emb = nn.Embedding(2, embed_dim)          # 0=user, 1=item

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb, self.item_emb, self.type_emb]:
            nn.init.xavier_uniform_(emb.weight)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        B = user_ids.size(0)
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        type_u = self.type_emb(torch.zeros(B, dtype=torch.long, device=user_ids.device))
        type_i = self.type_emb(torch.ones(B, dtype=torch.long, device=user_ids.device))

        seq = torch.stack([u + type_u, i + type_i], dim=1)   # (B, 2, D)
        encoded = self.transformer(seq)                      # (B, 2, D)
        pooled = encoded.mean(dim=1)                         # (B, D)
        return self.head(pooled).squeeze(-1)

# -------------------------------------------------
# 3. Architecture Visualisation (matplotlib)
# -------------------------------------------------
def visualise_architecture(save_path="uit_rec_architecture.png"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("User–Item Transformer Recommender (UIT-Rec)", fontsize=16, pad=20)

    def box(x, y, w, h, text, color, fontsize=9):
        fancy = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.2",
                               facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.9)
        ax.add_patch(fancy)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight="bold")

    # Inputs
    box(1.0, 8.5, 2.2, 0.8, "User ID", "#E0E0E0")
    box(4.0, 8.5, 2.2, 0.8, "Item ID", "#E0E0E0")

    # Embeddings
    box(0.8, 6.8, 2.6, 0.9, "User Embedding\n(d)", "#90CAF9")
    box(3.8, 6.8, 2.6, 0.9, "Item Embedding\n(d)", "#90CAF9")
    box(7.0, 6.8, 2.6, 0.9, "Type Embedding\n(user / item)", "#BBDEFB")

    # Sequence
    box(2.5, 5.2, 5.0, 0.8, "Sequence  [User, Item]   (B, 2, d)", "#CE93D8")

    # Transformer
    box(1.5, 3.0, 7.0, 1.6, "Transformer Encoder  (N layers)\nMulti-Head Self-Attention  +  FFN  +  Residual + LayerNorm", "#FFCC80")

    # Pool + Head
    box(2.5, 1.6, 3.0, 0.7, "Mean Pooling", "#80CBC4")
    box(6.0, 1.6, 3.5, 0.7, "MLP Head  (Linear → GELU → Dropout → Linear)", "#4DB6AC")
    box(5.0, 0.3, 3.0, 0.7, "Predicted Rating", "#81C784")

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    arrow(2.1, 8.5, 2.1, 7.7)
    arrow(5.1, 8.5, 5.1, 7.7)
    arrow(2.1, 6.8, 4.0, 6.0)
    arrow(5.1, 6.8, 5.5, 6.0)
    arrow(8.3, 6.8, 6.5, 6.0)
    arrow(5.0, 5.2, 5.0, 4.6)
    arrow(5.0, 3.0, 4.0, 2.3)
    arrow(5.0, 3.0, 7.5, 2.3)
    arrow(4.0, 1.6, 6.5, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Architecture diagram saved to: {save_path}")

# -------------------------------------------------
# 4. Quick demo (replace numbers with your real sizes)
# -------------------------------------------------
if __name__ == "__main__":
    # ----- CHANGE THESE -----
    num_users = 1000
    num_items = 500
    # ------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UserItemTransformer(
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Visualise
    visualise_architecture()

    # Dummy forward pass
    dummy_users = torch.randint(0, num_users, (4,)).to(device)
    dummy_items = torch.randint(0, num_items, (4,)).to(device)
    with torch.no_grad():
        preds = model(dummy_users, dummy_items)
    print("\nDummy predictions:", preds.cpu().numpy())
