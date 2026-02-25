import torch
import torch.nn as nn
from config import BertConfig


class BertBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, pad_mask=None):
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=pad_mask
        )

        x = self.attn_norm(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_output))

        return x


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_token = x[:, 0]
        return self.activation(self.dense(cls_token))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.word_embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_id
        )

        self.type_embed = nn.Embedding(
            config.num_types,
            config.hidden_size
        )

        self.pos_embed = nn.Embedding(
            config.max_seq_len,
            config.hidden_size
        )

        self.embeddings_norm = nn.LayerNorm(config.hidden_size)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)

        self.blocks = nn.ModuleList([
            BertBlock(
                config.hidden_size,
                config.num_heads,
                config.dropout_prob
            )
            for _ in range(config.num_layers)
        ])

        self.pooler = BertPooler(config.hidden_size)

    def forward(self, input_ids, token_type_ids):
        pad_mask = input_ids == self.word_embed.padding_idx

        batch_size, seq_len = input_ids.shape

        pos_ids = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        token_embed = self.word_embed(input_ids)
        type_embed = self.type_embed(token_type_ids)
        pos_embed = self.pos_embed(pos_ids)

        x = token_embed + type_embed + pos_embed
        x = self.embeddings_norm(x)
        x = self.embeddings_dropout(x)

        for block in self.blocks:
            x = block(x, pad_mask)

        pooled_output = self.pooler(x)

        return x, pooled_output


class BertPretrainingModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.bert = BertModel(config)

        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

        self.nsp_head = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids
        )

        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)

        return mlm_logits, nsp_logits