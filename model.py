import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)

        word = self.word_embeddings(input_ids)
        pos = self.position_embeddings(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)

        embeddings = word + pos + token_type
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()

        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        return context


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = BertSelfAttention(config)
        self.attention_output = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size)

        self.intermediate = nn.Linear(
            config.hidden_size,
            config.intermediate_size
        )

        self.output = nn.Linear(
            config.intermediate_size,
            config.hidden_size
        )

        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)

        hidden_states = self.output_norm(hidden_states + layer_output)

        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        all_layers = []

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_layers.append(hidden_states)

        return all_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        return self.activation(self.dense(cls_token))


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)

        all_layers = self.encoder(
            embedding_output,
            extended_mask
        )

        sequence_output = all_layers[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output, all_layers