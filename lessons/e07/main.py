import torch
import torch.nn as nn

class SingleHead(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        # attn layer
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(input_dim, attn_dim)
        self.k_proj = nn.Linear(input_dim, attn_dim)
        self.v_proj = nn.Linear(input_dim, attn_dim)

    def forward(self, x):
        token_size = x.shape[1]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2, -1) / (self.attn_dim ** 0.5)
        allowed = torch.tril(
            torch.ones(token_size, token_size, dtype = torch.bool, device = x.device)
        )
        masked = ~allowed
        masked_scores = scores.masked_fill(
            masked, float("-inf")
        )
        A = torch.softmax(masked_scores, dim = -1)
        output = A @ V

        return output

class MultiHead(nn.Module):
    def __init__(self, input_dim, attn_dim, output_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(input_dim, attn_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*attn_dim, output_dim, bias=True)
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        merged = torch.cat(outputs, dim=-1)
        outputs = self.proj(merged)

        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, attn_dim, output_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(input_dim)
        self.attn = MultiHead(input_dim, attn_dim, output_dim, num_heads)
        self.ln2 = nn.LayerNorm(output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 4*output_dim),
            nn.GELU(),
            nn.Linear(4*output_dim, output_dim)
        )

    def forward(self, x):
        u = x + self.attn(self.ln1(x))
        y = u + self.mlp(self.ln2(u)) # 这里的残差要加上u而不是x

        return y


torch.manual_seed(42)
def main():
    X = torch.randn(2, 4, 8)
    model1 = SingleHead(8, 4)
    model2 = MultiHead(
        input_dim = 8,
        attn_dim = 4,
        output_dim = 8,
        num_heads = 2
    )
    model3 = TransformerBlock(
        input_dim = 8,
        attn_dim = 4,
        output_dim = 8, 
        num_heads = 2
    )
    outputs1 = model1(X)
    outputs2 = model2(X)
    outputs3 = model3(X)
    print(outputs1.shape) # (2, 4, 4)
    print(outputs2.shape) # (2, 4, 8)
    print(outputs3.shape) # (2, 4, 8)


if __name__ == "__main__":
    main()
