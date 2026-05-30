import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Backbone(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.projection(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 256, temperature: int = 10000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_half = d_model // 2

        i = torch.arange(self.d_half // 2, dtype=torch.float32)
        self.register_buffer('freq', temperature ** (2 * i / self.d_half))

    def _compute_1d_pe(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions[:, None] / self.freq[None, :]
        pe = torch.zeros(len(positions), self.d_half, device=positions.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y_pe = self._compute_1d_pe(torch.arange(h, device=x.device, dtype=torch.float32))
        x_pe = self._compute_1d_pe(torch.arange(w, device=x.device, dtype=torch.float32))

        y_broadcast = y_pe[:, None, :].expand(h, w, self.d_half)
        x_broadcast = x_pe[None, :, :].expand(h, w, self.d_half)

        pe = torch.cat([y_broadcast, x_broadcast], dim=-1)
        return pe.permute(2, 0, 1).unsqueeze(0)   # (1, d_model, H, W)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.1, ffn_dim: int = 2048):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn1 = nn.Linear(embedding_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pe, src_key_padding_mask=None):
        q = k = x + pe
        v = x
        attn_out, _ = self.mhsa(q, k, v, need_weights=False,
                                 key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn2(self.dropout(self.relu(self.ffn1(x))))
        return self.norm2(x + self.dropout(ffn_out))


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, dropout, ffn_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, pe, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pe, src_key_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int = 256, d_ff: int = 2048,
                 dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.ffn1 = nn.Linear(embedding_dim, d_ff)
        self.ffn2 = nn.Linear(d_ff, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, pe, query_pos, memory_key_padding_mask=None):
        # self-attention on object queries
        q = k = tgt + query_pos
        attn1, _ = self.self_attn(q, k, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout(attn1))

        # cross-attention: queries attend to encoder memory
        q2 = tgt + query_pos
        attn2, _ = self.cross_attn(q2, memory + pe, memory,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=False)
        tgt = self.norm2(tgt + self.dropout(attn2))

        # FFN
        ffn_out = self.ffn2(self.dropout(self.relu(self.ffn1(tgt))))
        return self.norm3(tgt + self.dropout(ffn_out))


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, d_ff: int = 2048, num_heads: int = 8,
                 dropout: float = 0.1, num_queries: int = 100, num_layers: int = 6):
        super().__init__()
        self.object_queries = nn.Parameter(torch.randn(num_queries, embedding_dim))
        self.query_pos = nn.Parameter(torch.randn(num_queries, embedding_dim))
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, d_ff, dropout, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, memory, pe, memory_key_padding_mask=None):
        B = memory.shape[1]
        tgt = self.object_queries.unsqueeze(1).expand(-1, B, -1).contiguous()
        query_pos = self.query_pos.unsqueeze(1).expand(-1, B, -1).contiguous()

        intermediate = []
        for layer in self.layers:
            tgt = layer(tgt, memory, pe, query_pos, memory_key_padding_mask)
            intermediate.append(tgt)

        return torch.stack(intermediate)   # (num_layers, num_queries, B, embedding_dim)


class FFNs(nn.Module):
    def __init__(self, num_classes: int = 80, hidden_dim: int = 256, embedding_dim: int = 256):
        super().__init__()
        self.bbox_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.class_head = nn.Linear(embedding_dim, num_classes + 1)  # +1 for no-object class

    def forward(self, x):
        return self.bbox_head(x).sigmoid(), self.class_head(x)


class DETR(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.1, num_encoders: int = 6, num_decoders: int = 6,
                 num_queries: int = 100, num_classes: int = 80):
        super().__init__()
        self.backbone = Backbone(d_model=d_model)
        self.pe = PositionalEncoding2D(d_model=d_model)
        self.encoder = Encoder(embedding_dim=d_model, num_heads=num_heads,
                               ffn_dim=ffn_dim, dropout=dropout, num_layers=num_encoders)
        self.decoder = Decoder(embedding_dim=d_model, d_ff=ffn_dim, num_heads=num_heads,
                               dropout=dropout, num_queries=num_queries, num_layers=num_decoders)
        self.ffns = FFNs(num_classes=num_classes, hidden_dim=d_model, embedding_dim=d_model)

    def forward(self, images, mask):
        # (B, 3, H, W) -> (B, d_model, H', W')  where H' = H/32
        x = self.backbone(images)
        H_prime, W_prime = x.shape[2], x.shape[3]

        # downsample padding mask to backbone spatial resolution
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(H_prime, W_prime), mode='nearest'
        ).squeeze(1).bool()                          # (B, H', W')
        flat_mask = mask_down.flatten(1)             # (B, H'*W')

        pe = self.pe(x)                              # (1, d_model, H', W')
        x    = x.flatten(2).permute(2, 0, 1)        # (H'*W', B, d_model)
        pe   = pe.flatten(2).permute(2, 0, 1)       # (H'*W', 1, d_model)

        memory = self.encoder(x, pe, flat_mask)                        # (H'*W', B, d_model)
        hs     = self.decoder(memory, pe, flat_mask)                   # (6, 100, B, d_model)

        bbox_pred, class_pred = self.ffns(hs)
        # bbox_pred:  (6, 100, B, 4)   — normalised [cx, cy, w, h], sigmoid-bounded
        # class_pred: (6, 100, B, 81)  — raw logits, no-object is class index 80
        return {
                'pred_logits': class_pred[-1].permute(1, 0, 2),   # (B, 100, 81) — final layer
                'pred_boxes':  bbox_pred[-1].permute(1, 0, 2),    # (B, 100, 4)  — final layer
                'aux_outputs': [
                    {
                        'pred_logits': class_pred[i].permute(1, 0, 2),  # (B, 100, 81)
                        'pred_boxes':  bbox_pred[i].permute(1, 0, 2),   # (B, 100, 4)
                    }
                    for i in range(5)   # layers 0–4, layer 5 is already in the top-level keys
                ]
                }
