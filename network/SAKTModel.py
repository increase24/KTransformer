import torch
import torch.nn as nn
import numpy as np

MAX_SEQ = 180

class FFN(nn.Module):
    def __init__(self, state_size = 200, forward_expansion = 1, bn_size=MAX_SEQ - 1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads=8, dropout=0.1, forward_expansion=1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion=forward_expansion, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight


class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout=0.1, forward_expansion=1, num_layers=1,
                 heads=8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim) #seq len: max_seq-1
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)
        self.layers1 = nn.ModuleList(
            [TransformerBlock(embed_dim, forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.layers2 = nn.ModuleList(
            [TransformerBlock(embed_dim, forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.layers3 = nn.ModuleList(
            [TransformerBlock(embed_dim, forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, recall, question_ids):
        device = recall.device
        recall = self.embedding(recall)
        pos_id = torch.arange(recall.size(1)).unsqueeze(0).to(device)
        pos_embed = self.pos_embedding(pos_id)
        recall = self.dropout(recall + pos_embed)
        recall = recall.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        exercise = self.e_embedding(question_ids)
        exercise = exercise.permute(1, 0, 2)
        for layer in self.layers1:
            att_mask = future_mask(exercise.size(0)).to(device)
            exercise, att_weight = layer(exercise, exercise, exercise, att_mask=att_mask)
            exercise = exercise.permute(1, 0, 2)
        for layer in self.layers2:
            att_mask = future_mask(exercise.size(0)).to(device)
            recall, att_weight = layer(recall, recall, recall, att_mask=att_mask)
            recall = recall.permute(1, 0, 2)
        for layer in self.layers3:
            att_mask = future_mask(exercise.size(0)).to(device)
            recall, att_weight = layer(exercise, recall, recall, att_mask=att_mask)
            recall = recall.permute(1, 0, 2)
        recall = recall.permute(1, 0, 2)
        return recall, att_weight


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout= 0.1, forward_expansion=1, enc_layers=1,
                 heads=8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, max_seq, embed_dim, dropout, forward_expansion, num_layers=enc_layers)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight