import torch
import torch.nn as nn

def create_model():
    # torch.manual_seed(42)
    model = nn.Linear(1, 1)

    return model

def create_optimizer(model, lr = 0.1, momentum = 0.0, weight_decay = 0.0):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

    return optimizer

def grad_norm(model):
    grad_w = model.weight.grad
    grad_b = model.bias.grad

    sq_sum_w = grad_w.square().sum()
    sq_sum_b = grad_b.square().sum()

    norm = torch.sqrt(sq_sum_w + sq_sum_b)

    return norm

def train_one_window(
        model, 
        optimizer,
        loss_fn,
        x,
        y,
        scaler=None
):
    optimizer.zero_grad()
    use_amp = scaler is not None
    for micro_x, micro_y in zip(x.chunk(2), y.chunk(2)):
        with torch.autocast(
            "cuda",
            dtype = torch.float16,
            enabled=use_amp
        ):
            pred = model(micro_x)
            loss = loss_fn(pred, micro_y) / 2

        if use_amp:
            scaler.scale(loss).backward() # 放大loss
        else:
            loss.backward()

    norm_scaled = grad_norm(model)
    if use_amp:
        scaler.unscale_(optimizer) # 还原梯度，并检查非有限值

    norm_before = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.1)
    nrom_after = grad_norm(model)

    if use_amp:
        scaler.step(optimizer) # 梯度有效时更新参数，否则可能跳过
        scaler.update() # 调整后续使用的缩放因子
    else:
        optimizer.step()

    results = {
        "norm_scaled": norm_scaled,
        "norm_before": norm_before,
        "norm_after": nrom_after,
        "dtype": model.weight.dtype,
        "loss dtype": loss.dtype,
        "prediction dtype": pred.dtype
    }

    return results

device = "cuda"

X = torch.tensor([-2, -1, 1, 2], dtype = torch.float32)
X = X.reshape(4, 1)
Y = 2 * X + 1

X = X.to(device)
Y = Y.to(device)

model_fp32 = create_model().to(device)
model_amp = create_model().to(device)
model_amp.load_state_dict(model_fp32.state_dict()) # 保证实验公平，两个model的初始weight和bias都是一样的

loss_fn = nn.MSELoss()
optimizer_fp32 = create_optimizer(model_fp32)
optimizer_amp = create_optimizer(model_amp)

scaler = torch.amp.GradScaler(
    "cuda",
    init_scale = 128.0
)

result_fp32 = train_one_window(
    model_fp32, optimizer_fp32,
    loss_fn, X, Y
)
print(result_fp32)
result_amp = train_one_window(
    model_amp, optimizer_amp,
    loss_fn, X, Y,
    scaler = scaler
)
print(result_amp)

max_diff = max(
    (p1 - p2).abs().max().item()
    for p1, p2 in zip(
        model_fp32.parameters(),
        model_amp.parameters()
    )
)

print(max_diff)

