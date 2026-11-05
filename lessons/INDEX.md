# 课程代码索引

2026-10-23 起按课程 ID 整理：`lessons/<课程ID>/` 存放代码，`lessons/<课程ID>/outputs/` 存放检查点、数据缓存、指标日志和图表。目录名使用小写，原有脚本名保留；多文件课程沿用其模块组织。

以下表格是本次迁移对照，不代表新的验收结果；实际学习状态以根目录 `PROGRESS.md` 为准。历史 Session Log 中的路径保持当时记录，可按本表定位。

| 课程 | 内容 | 原路径（相对仓库根目录） | 当前代码（相对本索引） |
|---|---|---|---|
| C01 | Tensor、索引与属性 | `lessons/01_tensor.py` | [c01/01_tensor.py](c01/01_tensor.py) |
| C02 | Shape、轴与存储布局 | `lessons/02_shape.py` | [c02/02_shape.py](c02/02_shape.py) |
| C03 | 广播与矩阵乘法 | `lessons/03_broadcast_matmul.py` | [c03/03_broadcast_matmul.py](c03/03_broadcast_matmul.py) |
| C04 | 设备与原地操作 | `lessons/04_device_inplace.py` | [c04/04_device_inplace.py](c04/04_device_inplace.py) |
| C05 | Autograd 与计算图 | `lessons/05_autograd.py` | [c05/05_autograd.py](c05/05_autograd.py) |
| C06 | 梯度累积与梯度模式 | `lessons/06_grad_control.py` | [c06/06_grad_control.py](c06/06_grad_control.py) |
| C07 | Parameter 与 Module | `lessons/07_module.py` | [c07/07_module.py](c07/07_module.py) |
| C08 | 线性回归训练循环 | `lessons/08_train_linear.py` | [c08/08_train_linear.py](c08/08_train_linear.py) |
| C09 | Dataset 与 DataLoader | `lessons/09_dataset_loader.py` | [c09/09_dataset_loader.py](c09/09_dataset_loader.py) |
| C10 | 初始化与模型状态恢复 | `lessons/10_state_dict.py` | [c10/10_state_dict.py](c10/10_state_dict.py) |
| C11 | CPU 调试、gradcheck 与计时 | `lessons/11_debug_check.py` | [c11/11_debug_check.py](c11/11_debug_check.py) |
| C11 | CUDA 显存与基准计时补充 | `lessons/11_cuda_check.py` | [c11/11_cuda_check.py](c11/11_cuda_check.py) |
| D01 | 多层感知机 | `lessons/12_mlp.py` | [d01/12_mlp.py](d01/12_mlp.py) |
| D02 | 优化器、衰减与调度器 | `lessons/13_optimizers.py` | [d02/13_optimizers.py](d02/13_optimizers.py) |
| D03 | 分类与交叉熵 | `lessons/14_classification.py` | [d03/14_classification.py](d03/14_classification.py) |
| D04 | 数据划分与泛化 | `lessons/15_generalization.py` | [d04/15_generalization.py](d04/15_generalization.py) |
| D05 | 归一化与 Dropout | `lessons/16_norm_dropout.py` | [d05/16_norm_dropout.py](d05/16_norm_dropout.py) |
| D06 | 卷积与池化 | `lessons/17_conv_pool.py` | [d06/17_conv_pool.py](d06/17_conv_pool.py) |
| D07 | 残差 CNN | `lessons/18_cnn_residual.py` | [d07/18_cnn_residual.py](d07/18_cnn_residual.py) |
| D08 | 完整 checkpoint 与续训 | 原规划为 `lessons/19_checkpoint_resume.py`，学生实际使用多模块 | [d08/](d08/)：`config.py`、`data.py`、`model.py`、`engine.py`、`checkpoint.py`、`compare.py`、`train.py` |
| D09 | 梯度累积、梯度裁剪与AMP | 新课程，无历史迁移脚本 | [20_grad_accum_clip.py](d09/20_grad_accum_clip.py)；[21_amp.py](d09/21_amp.py) |
| D10 | Shape、梯度与SGD更新检查 | 新课程，无历史迁移脚本 | [main.py](d10/main.py) |

## 运行与产物

从仓库根目录运行历史脚本，例如：

```sh
conda run -n dllab python lessons/c10/10_state_dict.py
```

C10、D04、D07 的文件读写已改为以脚本目录定位 `outputs/`，从其他工作目录启动也使用同一产物位置。D08 已有 `config.py` 管理本课路径；模块草稿保留，尚未验收。

本次已迁移的现有产物：

| 原路径 | 当前路径 |
|---|---|
| `lessons/model_1.pt` | `lessons/d04/outputs/model_1.pt` |
| `lessons/model_2.pt` | `lessons/d04/outputs/model_2.pt` |
| `lessons/best_model.pt` | `lessons/d07/outputs/best_model.pt` |

C10 下次执行时将读写 `lessons/c10/outputs/linear_c10.pt`；本次未发现或生成该文件。D08 运行产物统一写入 `lessons/d08/outputs/`。

每个 `outputs/` 保留 `.gitkeep`，以便 Git 保留空目录。现有 `.gitignore` 继续忽略 `*.pt` / `*.pth`，因此上述已存在的模型文件仅在本机迁移，不会随普通 Git 提交同步。代码和文档可按正常流程一同提交。
