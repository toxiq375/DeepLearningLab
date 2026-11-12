import torch
import torch.nn as nn

class SingleHead(nn.Module):
    def __init__(self, input_channels, attn_channels):
        super().__init__()
        self.attn_channels = attn_channels
        self.q_proj = nn.Linear(input_channels, attn_channels, bias=False)
        self.k_proj = nn.Linear(input_channels, attn_channels, bias=False)
        self.v_proj = nn.Linear(input_channels, attn_channels, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        token_size = x.shape[1]
        scores = Q @ K.transpose(-2, -1) / (self.attn_channels ** 0.5)
        mask_allowed = torch.tril(torch.ones(token_size, token_size, dtype=torch.bool, device=x.device))
        mask_blocked = ~mask_allowed
        blocked_scores = scores.masked_fill(mask_blocked, float("-inf"))
        A = torch.softmax(blocked_scores, dim=-1)
        output = A @ V

        return output

class MultiHead(nn.Module):
    def __init__(self, input_channels, attn_channels, output_channels, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(input_channels, attn_channels) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*attn_channels, output_channels) # 记得这里的输入是num_heads * attn_channels,因为多个Head的输出进行了拼接

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.proj(outputs)

        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, input_channels, attn_channels, output_channels, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_channels)
        self.attn = MultiHead(input_channels, attn_channels, output_channels, num_heads)
        self.ln2 = nn.LayerNorm(output_channels)
        self.mlp = nn.Sequential(
            nn.Linear(output_channels, 4*output_channels),
            nn.GELU(),
            nn.Linear(4*output_channels, output_channels)
        )

    def forward(self, x):
        u = x + self.attn(self.ln1(x))
        y = u + self.mlp(self.ln2(u))

        return y

class miniGPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_size,
            embed_dim,
            input_channels,
            attn_channels,
            output_channels,
            num_heads,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_size, embed_dim)
        self.attn_block = TransformerBlock(input_channels, attn_channels, output_channels, num_heads)
        self.ln_f = nn.LayerNorm(output_channels)
        self.lm_head = nn.Linear(output_channels, vocab_size)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        emb = token_emb + pos_emb
        outputs = self.lm_head(self.ln_f(self.attn_block(emb)))

        return outputs
