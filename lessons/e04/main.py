import torch
import torch.nn as nn

torch.manual_seed(42)
x = torch.randn(2, 4, 8)

torch.manual_seed(42)

class SingleHead(nn.Module):
    def __init__(self, feat_dim, attn_dim):
        super().__init__()
        # 三个投影层
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, attn_dim, bias=False)

    def forward(self, x):
        # x:[B, T, C]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        token_size = x.shape[1]
        scores = Q @ K.transpose(-2, -1) / (self.attn_dim ** 0.5)
        allowed = torch.tril(
            torch.ones(token_size, token_size, dtype=torch.bool, device=x.device)
        )
        blocked = ~allowed
        masked_scores = scores.masked_fill(
            blocked, float("-inf")
        )
        A = torch.softmax(masked_scores, dim=-1)
        
        output = A @ V

        return output, A

def main():
    X = torch.randn(2, 4, 8)
    model = SingleHead(8, 4)

    output, weights = model(X)
    loss = output.square().mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        print(name, "Gradient check: Passed")

    print("Output shape:", output.shape)
    print("Weights shape:", weights.shape)
    print("Weights:", weights)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6
    )

    X_changed = X.detach().clone()
    X_changed[0, -1, -1] = 1000.0
    output_changed, weights_changed = model(X_changed)
    print("Weights:", weights_changed)

    assert not torch.allclose(
        output[0:, 3, :],
        output_changed[0:, 3, :],
        atol = 1e-6
    )

if __name__ == "__main__":
    main()
