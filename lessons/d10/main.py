# 常见的深度学习错误检测，全当复习一遍
import torch
import torch.nn as nn

def check_shape(pred, tgt):
    assert pred.shape == tgt.shape, \
        f"Shape msimatch: {pred.shape} vs {tgt.shape}"

torch.manual_seed(42)

X = torch.tensor([[-2.], [-1.], [1.], [2.]])
Y = 2*X + 1

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
criterion = nn.MSELoss()

def main():
    pred = model(X)
    check_shape(pred, Y)
    wrong_Y = Y.squeeze(dim = -1)
    # check_shape(pred, wrong_Y)

    try:
        check_shape(pred, wrong_Y)
    except AssertionError:
        print("Shape Error Found")
    else:
        raise AssertionError("No Shape Error Found")

    for epoch in range(5):
        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(pred, Y)
        loss.backward()

        assert model.weight.grad is not None

        w_before = model.weight.detach().clone()
        g_before = model.weight.grad.detach().clone()
        w_expected = w_before - 0.1 * g_before

        assert model.weight.grad.shape == model.weight.shape
        assert torch.isfinite(model.weight.grad).all()
        assert torch.any(model.weight.grad != 0)

        optimizer.step()

        torch.testing.assert_close(
                model.weight.detach(),
                w_expected,
                rtol = 1e-5,
                atol = 1e-6
            )

if __name__ == "__main__":
    main()
