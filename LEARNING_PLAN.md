# DeepLearningLab：从 Python 初学者到独立实现现代深度学习算法

> 版本：v1.1（2026-10-28 目标与节奏复核；保留既有课程索引和历史）
> 文件作用：**长期课程总纲 + 课程验收标准 + 跨 Agent 交接依据**。  
> 配套文件：`AGENTS.md`（常驻约束）、`PROGRESS.md`（唯一实际进度）、`.agents/skills/deep-learning-mentor/SKILL.md`（教学工作流）、`docs/MENTOR_GUIDE.md`（备课/审阅/记录操作）、`docs/RESOURCE_MAP.md`（作者资料与课程映射）。
> 项目根目录以当前工作区与设备实测为准；Windows 为 `C:\Github\projects\DeepLearningLab`，Linux 路径按 PROGRESS.md 的设备记录核实，不因原始路径示例自动移动仓库。

## 0. 读者说明：请新 Agent 先看这里

本项目是一门以**学生亲手写代码**为核心的长期、个性化编程课程。学生明确自述为计算机专业研究生，希望以原理与编程结合的方式入门深度学习。具体 Python、数学和 PyTorch 基础以实际作业及学生反馈判断，不因学历预设熟练，也不把旧阶段的困难当作永久能力标签。AI 提供目标、提示、API 说明和审阅，由学生完成核心实现。

**最终学习目标不是“跑通一个教程”，而是在不用 AI 代写核心算法的条件下：**

1. 自己设计、编写、调试并维护 Python/PyTorch 研究代码。
2. 看懂陌生深度学习项目的模块边界、数据流、Tensor Shape、数学公式、训练/采样过程。
3. 从空仓库实现并训练可运行的 mini-GPT / nanoGPT 风格语言模型。
4. 从零实现 Transformer、VAE、GAN、Normalizing Flow、Diffusion（DDPM/DDIM/score 视角）、Flow Matching / Rectified Flow 等核心算法。
5. 迁移这些能力到视觉、三维、动态人体、3D Gaussian Splatting、生成式建模等研究代码（选修专题，不抢占基础课程）。
6. 能独立定位报错、复现实验、做消融、为代码编写测试并评估是否符合论文原理。

**特别注意：**截至本计划建立时，学生仅在先前对话中**听过**第 1 节课的说明，尚未提交作业，也未验证其掌握情况。不能把“讲过”当作“已通过验收”。具体状态以 `PROGRESS.md` 为准。

### 2026-10-28 目标复核：后续教学的优先解释

目标与方法总体一致，保留已通过成果；本次调整解决过长前置工程、课题碎片化、旧交接快照误导等问题。当前优先路线为“已有神经网络基础 → D09 累积/裁剪 → D10 与现有成果的必要验证 → 字符语言模型 → 因果注意力 → mini-GPT训练/采样 → nanoGPT源码对照”。D08 CUDA最终复核和D09 CUDA AMP实验保留为明确待补事项，不阻塞独立的CPU主题。

- 每个主题有数学问题、手写代码产物和少量核心验证；具体安排见下方E阶段主题课表。主题不是一次回复必须写完的代码量，可以按学生反馈分几次讲解和实现。
- mini-GPT先用字符词表、小模型和小文本，做到训练/验证/生成/恢复可解释。完整BPE、RoPE、KV Cache、FlashAttention、DDP及复现GPT-2规模是后续扩展，不作首个模型的前置门槛。
- P-D以已有D阶段实现综合核对；不为了文件名、模块数或测试个数再写一个相同分类工程。真正影响数学/数据/更新正确性的错误仍需学生修正。
- 每次备课选与本主题直接对应的作者章节/练习及当前官方API，不要求按清单先看完全部视频或slides；教材旧代码与本机版本不一致时解释差异，不能照抄安装或性能结论。
- 参考资料核实范围与链接见 `docs/RESOURCE_MAP.md`；本次检查结论见 `docs/LEARNING_AUDIT_2026-10-28.md`。这些是路线与审查记录，实际下一步只读 `PROGRESS.md`。

## 1. 真实环境和现有项目状态（2026-10-10）

### 1.1 用户实际环境（学生自述）

- 系统：Ubuntu / Linux；本地有 NVIDIA GPU（此前交流中使用过约 96 GB 显存的 RTX PRO 6000 级别设备；**Agent 不得据此假定本次机器状态，使用前实测**）。
- CUDA 方向：12.8。
- Conda 环境：**`dllab`**（不是最早教学示例中的 `learn-gpt`）。
- Python：**3.11**。
- PyTorch：**2.10.0**。
- 使用本地 Codex 进行大部分课程，也会返回网页版 ChatGPT 讨论。
- 已有目录：

```text
~/projects/DeepLearningLab/
├── lessons/
├── nanoGPT/
└── README.md
```

- `nanoGPT/` 的现有内容、Git 状态、环境实际可用性都**尚未由导师验证**；开始前只读检查，**不得覆盖或重置**。

2026-10-28目录核对：当前Windows工作区未发现 `nanoGPT/`，上面的目录树为建立计划时的历史记录；不据此自动克隆或创建。将来按需引入参考仓库，学生手写实现放在独立项目位置。

### 1.2 环境操作约定

- 首次仅检查（无需重新创建）：`conda info --envs`、`python --version`、`python -m pip show torch`。
- 按需检查 CUDA：`python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'`。
- 确认 `which python` / `sys.executable` 指向 `dllab`，理解“驱动 CUDA / Toolkit / PyTorch CUDA wheel”的区别。
- **除非学生明确同意，否则不要安装/升级/降级包、创建第二个 Conda 环境、清理文件或覆盖已有模型权重。**
- 环境不兼容时，先给出证据、原因、最小修复方案，再让学生执行；不要无缘无故修改 CUDA、Torch 版本。
- GPU 大显存只影响可运行的实验规模，不应跳过基础训练。默认使用 CPU 或很小的 GPU 张量学习，逐步扩大。

### 1.3 目录组织（课程目录于2026-10-23更新，其他项目按需扩展）

```text
DeepLearningLab/
├── AGENTS.md                         # Codex 常驻规则（本包提供）
├── LEARNING_PLAN.md                  # 本文：课程路线（本包提供）
├── PROGRESS.md                       # 学习记录（本包提供）
├── .agents/skills/deep-learning-mentor/SKILL.md
├── README.md                         # 学生已有；不覆盖
├── lessons/
│   ├── c01/
│   │   ├── 01_tensor.py              # 保留学生原脚本名
│   │   └── outputs/                  # 本课运行产物
│   ├── c11/                         # 同课可有 CPU/CUDA 多份脚本
│   ├── d01/
│   │   ├── 12_mlp.py
│   │   └── outputs/
│   ├── d08/                         # 已有多模块课程
│   │   ├── train.py                  # 入口，其他模块与它同层
│   │   ├── ...
│   │   └── outputs/
│   ├── INDEX.md                     # 课程与迁移路径对照
│   ├── ...
│   └── notes/                        # 可选：学生自己的笔记
├── nanoGPT/                          # 学生已有；先检查后再决定如何使用
├── projects/                          # 后续：从头实现的算法项目
│   ├── mlp_from_scratch/
│   ├── vision_classifier/
│   ├── mini_gpt/
│   ├── vae/
│   ├── ddpm/
│   └── flow_matching/
├── paper_reading/                    # 后续：论文公式到代码的映射
└── tests/                            # 后续：算法正确性测试
```

除学生已授权整理的现有课程外，**不要提前创建所有未来项目目录**；课程进展到对应专题时，学生应亲自解释为何引入目录并创建它。`nanoGPT/` 和未来的 `projects/mini_gpt/` 可分别用作参考阅读和本人手写实现（若已有内容不符，先协商命名）。

### 1.4 课程代码与产物约定（2026-10-23 学生更新）

统一使用 `lessons/<小写课程ID>/` 存放本课代码、其 `outputs/` 存放检查点/缓存/日志/图表；多份练习或模块可同属一课，保留学生原有脚本名与实现。课程与旧路径的完整对照见 `lessons/INDEX.md`。代码按文件所在位置定位产物，不依赖启动工作目录；D08已有config.py集中管理路径。`outputs/.gitkeep`用于Git保留空目录，模型文件继续遵守现有忽略规则。当前规划使用新路径；历史进度Session保留原路径与当时事实，通过索引追溯。目录迁移不代表重跑或重新验收，不改变本章算法要求。

## 2. 核心教学契约（所有 Agent 必须遵守）

### 2.1 教学原则

1. **学生是代码作者，Agent 是导师、审阅者和调试教练。**默认不直接提供完整、可复制粘贴的一整个实现文件或完整实验项目。
2. 按学生 2026-10-13 明确的新偏好授课：**本章节内容与意义 → 原理/数学/运行机制 → Python/PyTorch API 的用法与适用场景 → 任务意义与目标 → 学生手写 → 集中反馈**。开章先说明解决什么实际编程问题、为什么需要学及如何联系后续算法。
3. 默认不提供可直接运行的代码，包括局部完整示例；使用文字、公式、Shape 表、接口签名及不可直接执行的伪代码。说明 API 参数、返回值、副作用及用法，由学生自己实现。仅在学生明确另行要求参考实现时调整。

   2026-10-13 学生补充要求：适当举例并详细解释抽象概念，不能只列术语或一句定义。采用“明确含义 → 具体数值/公式/图示 → 对照判断 → 实际用途与常见误解”的方式；如 is_leaf 与 grad_fn，应展示直接创建的输入和被追踪运算结果的区别，以及它们与梯度保存的联系。学生反馈未理解时先补讲，不额外增加验收门槛；示例不默认包含可运行代码。
4. **每次给出一个范围适当的完整任务**。按学生 2026-10-12 的新偏好，把相关概念合并为完整课块讲解、集中练习与验收，减少每个 API 单独往返；遇到明确障碍再拆分，不一次堆积无关知识。
5. 错误反馈先指出位置、现象、假设和检查手段；必要时逐级加提示，不上来粘贴修改后的完整代码。
6. 每个 Python 关键语法第一次出现，解释其含义与常见用法：例如 `*args`、`**kwargs`、继承、`super()`、`@property`、`@staticmethod`、`@torch.no_grad()`、`__call__`、上下文管理器、类型注解等。
7. 对每一个核心 Tensor 运算回答：**输入 Shape / 输出 Shape / dtype / device / 是否参与计算图**；出现 `reshape`、`transpose`、`permute`、`matmul`、广播时尽量用小尺寸算例说明。
8. 对模型与算法回答：**问题定义 → 目标函数 → 参数与数据 → 前向传播 → 损失计算 → 反向传播 → 推理或采样 → 验证方式**。
9. 明确区分“学生能复述”“学生能修改”“学生能独立写出并解释”；只有通过验收才标记 mastered。
10. 尊重学生的真实代码，避免未经允许的大规模重构、文件覆盖、偷偷补全答案。
11. 语气友好、细致、不居高临下；不因卡在基础知识而跳过讲解。
12. 追求可迁移能力：同一概念分别在语言、图像和生成模型里认出来。

### 2.2 推荐的 Socratic 提示阶梯

- L0：复述要求和输入/输出，不给实现。
- L1：指明应查阅的 API/语法、数学关系或相关文件。
- L2：不可直接执行的伪代码/接口签名、Shape 表、文字描述的局部操作。
- L3：指出具体 Bug 及局部修改策略。
- L4：仅在用户明确另行要求代码参考时提供局部参考，并解释为什么；**不要伪装成学生独立完成**。多次受阻时先补充原理与用法，不自动突破默认“不提供可直接运行代码”的偏好。

学生随时可选择“只要提示 / 详细讲解 / 代码审阅 / 参考答案”，默认 L1–L2；遇到硬性阻碍（驱动/安装失败）可用完整**命令**协助，但核心训练/算法代码仍保持教学优先。

### 2.3 每节课的固定流程

1. **开课前**：每次开始或恢复课程，重新读取磁盘上的 `AGENTS.md`、`PROGRESS.md` 当前状态与最近实际 Session Log，以及本计划对应模块；确认章节、验收状态和最小任务，只读观察所需代码。切换设备或 Git pull 后同样执行，不凭聊天记忆跳课。
2. **开章介绍与必要复习**：先说明本章内容、学习意义、实际用途及与前后章节的联系；有明确前置缺口时直接提问，不机械重复已通过内容。
3. **讲解与用法**：解释概念、数学、Shape 与机制；逐项说明 API 参数、返回值、副作用、适用场景及操作方法，不给可直接运行的代码。
4. **布置任务**：首先说明任务的意义、能力目标和完成后能解决什么问题，再明确文件路径、输入输出、核心验收条件及参考 API；不默认要求把预测或答案写成注释。2026-10-17 学生补充：同时给出按依赖顺序组织的实现框架，每个功能块说明目的、输入/输出与 Shape/dtype/device、具体操作、API 参数/返回值/副作用及使用位置、核心自查点；附主流程伪代码、必要的函数接口或记录字段，明确第一步与必做/可选项。不能只列术语/API 名称，也不把框架变成完整可运行答案；函数化为组织建议，不新增强制测试、预测注释或验收关卡。
5. **学生编写**：鼓励学生先自己写并运行，不替代键盘。
6. **代码 Review**：先验证实际行为，再逐项解释问题；引导学生自行修复。
7. **验收**：代码在当前教学环境运行正常、结果正确并体现本章核心内容即可通过。核心算法/数学/数据流错误需修正；非核心兼容性、打印/命名/格式等作为改进建议，不反复阻塞进度。有明确理解疑点时直接提出针对性问题，学生自由选择聊天或注释回答；不强制每课边界测试、变体或整张量预测。
8. **记录**：学生已于 2026-10-12 持续授权自动记录。每次章节/小节/作业验收后立即更新 `PROGRESS.md`，无需再次询问是否记录；含日期、文件、实际证据、结论、提示依赖与下一步。只有真实通过才标记 `PASSED`；未通过记录剩余任务。路线或规则改变时同步维护对应指导文档。建议将作业与记录一起提交 Git。

### 2.4 验收维度（统一评分建议）

每次作业从 0–2 分评估：

| 维度 | 0 | 1 | 2 |
|---|---|---|---|
| 语法与 API | 不理解 | 借助提示完成 | 独立使用并解释 |
| 数学与 Shape | 不清楚 | 部分说清 | 能手算/追踪变形 |
| 正确性 | 失败 | 主路径成功 | 边界与小测试成功 |
| 调试能力 | 依赖 Agent 改 | 会定位 | 能建立假设并验证 |
| 工程质量 | 混乱 | 可运行 | 可读、模块化、可复现 |

**用途与通过标准（2026-10-13 更新）**：本表用于诊断与安排复习，不作为逐项满分的硬性通过门槛。基础章节以代码可运行、结果正确且体现核心知识为通过标准；无提示独立熟练度、工程细节及非核心边界完善程度分别记录。项目阶段以明确的核心功能和算法正确性验收，不以无关细节阻塞。真实掌握证据仍须保留，不能把运行成功等同于全面熟练。

## 3. 课程路线图与依赖关系

### 当前采用的授课方式（2026-10-12 学生确认）

以下路线描述知识依赖，实际教学以 **C 阶段 PyTorch 及后续深度学习实践为主线，A/B 按需穿插并与实践联系**，不机械要求先完整重学 A/B 再进入 C。

学生自述已了解 A/B 中许多概念，而深度学习编程与 PyTorch 实践经验相对不足，因此优先用实践组织课程。此项仅记录学习背景与偏好，不将“了解概念”认定为独立实现能力或完整验收通过。

- 熟悉的 Python/数学基础通过简短检查、小变体或实践解释确认，避免重复完整授课。
- 新 API 与算法显式联系其 Python 语法、数学原理、Shape 和工程用途；必要前置概念应在使用前讲清。
- 发现基础缺口时，插入范围明确的 A/B 课块再返回主线，不等到报错后才零碎补一句。
- A/B 课程内容与 P-A/P-B 验收项目保留；后续结合工程或算法实践安排验收，不因 C 阶段推进而自动升级 A/B 状态。
- 相关概念合并为完整课块，集中练习和运行验收；先定义术语与 API，再讲机制，避免只增加概念密度。

近期衔接示例：C03 联系 B03/B04（广播、向量化、点积与矩阵乘法）；C05 联系 B05（偏导数与链式法则）；C07 联系 A03/A05/A06（函数、类、继承与 super）；多文件训练工程联系 A07/A08/A12（包、路径、配置与入口）。具体进度以 PROGRESS.md 为准。

主线建议按顺序推进：

`以C/D实际作业为起点（Python/数学按需补）→ 字符语言模型 → 因果Transformer/mini-GPT → nanoGPT源码对照 → 按研究需求扩展VAE/GAN/Flow/Diffusion/Flow Matching`

**纵向训练**：工程、源码阅读和必要正确性检查随当前算法逐步引入；不另设必须先完成的工程化阶段。不要为了追赶日期跳过关键前置条件，也不因无关环境/风格问题反复扩张前置任务。

下列建议课次编号是**课程索引**，不是已存在的文件，也不是固定学时。每课视难度可拆成几次对话。

## 4. 阶段 A：Python 进阶与工程语法（A01–A12）

**目标**：不再把 Python 代码视为模板，而是理解语法、运行机制与可维护的组织方式。

| 课次 | 学习主题 | 学生亲手任务 / 迁移场景 |
|---|---|---|
| A01 | 基础回顾：对象、变量绑定、可变与不可变、`id`/浅拷贝 | 预测修改列表/张量变量后的效果 |
| A02 | 列表、元组、字典、集合；推导式、切片与解包 | 从嵌套训练配置中提取/整理参数 |
| A03 | 函数：默认参数、关键字参数、作用域、返回多个值 | 写参数合法性检查函数 |
| A04 | `*args`、`**kwargs`、匿名函数、闭包 | 理解模型初始化和配置传递 |
| A05 | 类与对象：`__init__`、实例/类属性、方法、`self` | 写简易 `MetricTracker` |
| A06 | 继承、`super()`、特殊方法、`@property` | 建立类层次；为 `nn.Module` 做准备 |
| A07 | 模块与包：`import`、相对/绝对导入、`__init__.py`、`__name__` | 把脚本改为可导入的多文件项目 |
| A08 | pathlib、文件读写、JSON/YAML 与路径 | 读取配置、处理日志文件 |
| A09 | 异常、断言、日志、调试、栈追踪 | 自己定位 3 种常见报错 |
| A10 | 迭代器/生成器、`yield`、`enumerate`、`zip` | 实现简单 batch 迭代器 |
| A11 | 上下文管理器、装饰器、类型标注、dataclass | 理解 `no_grad` 和配置类 |
| A12 | `argparse`、入口文件、环境/依赖、Git 基础、测试简介 | 可执行的多模块 CLI 小工程 |

**验收项目 P-A**：自建 `lessons/python_toolbox/`（名称可自定），实现小型数据读取 + 配置 + 统计 CLI。能解释 Python 包是如何被 import 的，`python file.py` 与 `python -m package.module` 有何不同。

**常见易错点**：默认参数放可变对象、`is` vs `==`、浅拷贝、循环变量作用域、循环 import、工作目录与路径、包的搜索路径。

## 5. 阶段 B：数学、NumPy 与张量思维（B01–B09）

**目标**：数学公式能落地成 Tensor Shape / 运算步骤，并具备数值直觉。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| B01 | 标量/向量/矩阵/张量、维度、轴 | 手写多个 Shape 与索引结果 |
| B02 | NumPy ndarray、切片、view/copy、连续存储 | 验证内存共享与修改影响 |
| B03 | 广播规则、批处理、向量化 | 用循环与广播实现同一算式 |
| B04 | 点积、矩阵乘法、转置、矩阵微积分 | 手算小矩阵与梯度 |
| B05 | 链式法则、梯度、Jacobian、偏导数 | 手推线性回归导数并做数值差分 |
| B06 | 概率、期望、方差、Gaussian、采样 | 模拟高斯分布的统计量 |
| B07 | 对数概率、最大似然、交叉熵、KL | 比较 NLL / CE / KL 含义 |
| B08 | 优化：SGD、Momentum、Adam、学习率 | 手算 1–2 步参数更新 |
| B09 | 数值稳定性：log-sum-exp、softmax、精度 | 对比朴素/稳定的 softmax |

**验收项目 P-B**：纯 NumPy 实现线性回归拟合（包括 loss 和手动梯度），与解析解或数值梯度核对；解释 batch 与维度。

## 6. 阶段 C：PyTorch 核心语言（C01–C11）

**目标**：能独立完成张量表达、设备管理、Autograd 和最小训练循环。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| C01 | Tensor 构造、dtype、device、shape、索引、切片 | **先完成初始作业 `lessons/c01/01_tensor.py`** |
| C02 | `reshape`、`view`、`permute`、`transpose`、`unsqueeze`/`squeeze` | 明确每一步维度变化 |
| C03 | 广播、`@`/matmul、`einsum` 入门 | 手算小尺寸批量矩阵乘法 |
| C04 | 设备、CPU↔CUDA、非原地与原地操作、`.to()` | 检测 device mismatch 并修复 |
| C05 | 计算图、`requires_grad`、叶子 Tensor、`backward` | 对比手算和 autograd |
| C06 | 梯度累积、`detach`、`no_grad`、`inference_mode` | 展示何时没有梯度 |
| C07 | `torch.nn.Parameter`、`nn.Module`、`forward`/`__call__` | 手写一个最小线性层类 |
| C08 | 损失函数、优化器、`zero_grad/backward/step` | 完成线性回归训练循环 |
| C09 | `Dataset`、`DataLoader`、batch、shuffle、collate | 实现自定义 Dataset |
| C10 | 初始化、随机种子、保存/加载 `state_dict` | 恢复权重后复现推理 |
| C11 | GPU 监测、内存、错误调试、基准测试、梯度检查 | 从错误 stack trace 定位问题 |

**C11 当前设备安排（2026-10-14）**：Windows CPU 环境先集中学习 traceback/shape 检查、非有限数值、Tensor 存储大小、有限差分与 gradcheck、CPU 小规模基准计时。综合练习使用已有线性回归公式，不重复训练/保存流程；CUDA 监测、显存分配与缓存及 GPU 异步计时在具备 CUDA 设备时补实践，不能由 CPU 作业推断已掌握。通过聚焦当前环境的核心调试与检查能力，不要求制造大量错误或长时间性能实验。

**C11 CUDA 补充与 P-C 核对（2026-10-14，恢复 Ubuntu 22.04/CUDA 12.8）**：保留 C11 CPU 通过及既有训练成果。P-C 的手写 Linear/MSE/SGD 已有 C08–C10 验收证据，补讲并确认 train/eval 与 no_grad 的区别，不要求另写完整训练项目。学生另建 `lessons/c11/11_cuda_check.py`，用两个 512×512 CUDA float32、无需梯度的 Tensor 做一次矩阵乘法；观察张量存储大小、memory_allocated/memory_reserved、释放输出与一次 empty_cache 的变化，再用已有 Timer 做短时 GPU 计算基准，理解异步执行与同步。验收聚焦正确含义与可运行实验，不要求特定显存数值或速度、不重复 CPU gradcheck；完成后集中核对 P-C，再由用户请求进入 D 阶段。

**验收项目 P-C**：只使用小规模 Tensor，自己写 `Linear` → `MSE` → `SGD` 最小训练流程；梯度与 loss 正确；解释 `model.train()` 与 `model.eval()` 的含义。

**C02 组织调整（2026-10-12，学生明确请求）**：将维度操作作为一个整体，按“增删单例轴 → 重组形状 → 交换/重排轴 → stride、连续性与 view/reshape”建立统一对照。已通过部分简短串联，剩余 permute 与 view/连续性合并讲解；集中布置一个综合练习、一次运行核对多个结果。学生仍亲手编写核心代码，理解确认并入综合验收，不逐 API 重复运行。后续课程沿用相关概念成组、集中验收的方式。

**关于最初的第 1 课（精确存档）**：先前只介绍过 `import torch`、`torch.tensor`、`.shape/.dtype/.device`、`x.mean()`、`x[0,1]`、`x[:,1]`、`mean(dim=...)`、`.to('cuda')`；布置了 2×3 浮点 Tensor、属性打印、选取行列、按列均值、移到 CUDA 的任务，含三个思考题：Shape `(2,3)` 的含义、双索引的含义、`x.to('cuda')` 是否原地修改。**没有提交/验收记录，仍是待完成**。

## 7. 阶段 D：从零训练神经网络（D01–D10）

**目标**：理解“模型类、损失、反向传播、优化器、数据、日志、验证”如何协同。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| D01 | 多层感知机 MLP、线性层、非线性、参数量 | 手写多层网络并计算参数量 |
| D02 | SGD/AdamW、权重衰减、调度器 | 对比收敛曲线 |
| D03 | 分类任务、CrossEntropyLoss、logits | 训练合成二维分类数据 |
| D04 | train/val/test、泛化、过拟合、数据泄露 | 设计对照实验 |
| D05 | BatchNorm/LayerNorm/Dropout | 区分训练和推理行为 |
| D06 | 卷积、padding、stride、pooling、输出形状 | 手写一个简化 Conv2D 运算思路 |
| D07 | CNN、ResNet、跳跃连接 | 构建图像分类器并解释残差 |
| D08 | 可重复性、checkpoint、日志、指标 | 从 checkpoint 恢复训练 |
| D09 | AMP 混合精度、梯度裁剪与累积 | 比较显存、速度、数值变化 |
| D10 | 单元测试、Shape 测试、梯度测试、过拟合单 batch | 编写能捕获实际 Bug 的测试 |

**D01 本轮组织（2026-10-15）**：C01–C11 既有代码验收保留，P-C 模式概念解释单独确认，不要求重复完整训练项目。先讲两层线性变换仍可合并的推导、ReLU 非线性、隐藏特征与参数共享，再手写 2→4→1 的 Module：四个 Parameter、隐藏层 ReLU、线性输出，联系 B03/B04 与链式求导。任务使用四个二维输入及乘积目标，核对批次/特征 Shape、参数总数、一次反向与一次 SGD 更新；本轮不增加数据加载/保存/计时或完整拟合门槛。初始化复用 C10 的随机权重与零偏置，学生自行实现。

**D02 本轮组织（2026-10-15）**：复用 D01 的 2→4→1 模型及四条回归数据，以四组同起点实验联系 SGD、AdamW 的历史梯度统计、解耦权重衰减和 StepLR。每组模型创建前重新设置种子 42，使用独立模型/优化器，统一初始 lr=0.05，每轮全量四样本一次更新，共 200 轮；依次比较 SGD(wd=0)、AdamW(wd=0)、AdamW(wd=0.1)、AdamW(wd=0.1)+StepLR(step_size=50,gamma=0.5)。记录训练前及每次更新后的全数据 MSE、该轮实际使用 lr 和最终参数 L2 范数，形成曲线数据并用表格比较，绘图为可选，不要求安装依赖。验收控制变量、参数更新、衰减配置、调度位置/频率和记录含义；不预设哪组获胜、不设低损失门槛、不据四点训练误差推断泛化，不要求手写 AdamW 内核或大规模调参。

**D03 本轮组织（2026-10-16）**：以三簇二维合成点完成互斥三分类，先说明类别编号与连续回归目标的区别，再联系 logits→softmax 概率→负对数似然、稳定的交叉熵计算和 argmax/accuracy；分类索引标签为 long [B]，原始 logits 为 float [B,3]，不在交叉熵前添加 softmax、argmax 或 detach。联系 B07，但不据此整模块验收。引入 nn.Linear，将已有手写 XW+b 对应到 X weight.T+b，解释权重存储 Shape 与子模块自动参数注册。任务为 lessons/d03/14_classification.py：种子 42，三个中心 (-2,-2)/(2,-2)/(0,2)，每类 64 点，独立标准正态噪声乘 0.5；拼接得 X=[192,2] float32、y=[192] long。学生自建 2→16→3 MLP，隐藏 ReLU、输出原始 logits，默认 Linear 初始化；整模型与数据统一 CUDA，之后创建 AdamW(lr=0.02,weight_decay=0)，全批量训练 200 次。记录训练前/后及每 50 次更新后重新计算的全数据 CE 与 accuracy，最终展示少量概率、预测/真实类别与概率行和。本轮训练指标只描述已见数据拟合，train/val/test 与泛化实验留待 D04；不要求 DataLoader、绘图、指定精度门槛或手写损失内核。验收聚焦正确的数据/标签、分类输出与损失语义、梯度更新和指标计算，由学生亲写实现。

**D04 本轮组织（2026-10-17）**：承接 D03 的训练集 100% 准确率，讲解经验训练误差、未见同分布数据的泛化估计、过拟合/欠拟合、train/val/test 的不同信息用途与数据泄露。用同一固定分层划分比较网络宽度，不强求更宽模型过拟合或某组获胜。任务为 lessons/d04/15_generalization.py：数据种子 42，沿用三中心与每类 64 点，将噪声系数从 0.5 改为 1.2；每类用一次 randperm(64) 划为 12/26/26 个 train/val/test，总数为 36/78/78，保留全局样本索引核对不交叠，X/y 同索引取样，划分后固定。两独立模型 A=2→8→3、B=2→128→3，各一层隐藏 ReLU、输出 logits；每组创建前 seed=123，模型 Shape 不同不要求初始参数相同，使用默认初始化。相同数据、AdamW(lr=0.01,weight_decay=0)、全训练集每轮一次更新、400 轮；只有 train 进入 backward/step。epoch 0 与每 20 次更新后，用当前参数 eval/no_grad 重算 train/val 的 CE 和 accuracy，记录 Python 数值。各组根据已评估轮次的最低 val CE 保留 best_epoch/best_val_loss 和独立参数快照（deepcopy(state_dict) 或逐张量 clone），第 400 轮指标单独保留。两组训练和验证选择均完成后，按最低 best_val_loss 选模型（相等时选 A），恢复该组最佳状态，最后仅评估选中模型的 test 一次，随后不据测试结果调参。说明 state_dict 的共享存储、load_state_dict 的原地恢复和返回值，不要求落盘或恢复优化器；D08 再处理完整 checkpoint。验收固定互斥划分、无验证/测试泄露、正确指标/最佳状态与验证选择协议，不设精度门槛，不要求绘图、DataLoader、归一化或特定曲线形状；学生用实际 train/val 趋势作简短解释，不能以一次差距或单次种子推广模型优劣。

**D04 实现框架提示（2026-10-17，原任务范围不变）**：以下功能块可写成函数，也可按脚本段落组织；两个现有模型类可保留，不要求重构。

| 实现顺序 | 功能与输入/输出 | 关键 API 与放置位置 | 核心自查 |
|---|---|---|---|
| 1. 数据生成 | 三个中心各 64 点，拼接 X=[192,2] float32、y=[192] long；先在 CPU 完成数据和索引准备 | 生成前 manual_seed(42)；randn 生成 [64,2] 噪声乘1.2加中心；full 生成 [64] 类别索引；cat 沿样本轴拼接 | 三类各64点，不是192点共用一个中心；X/y配对 |
| 2. 固定分层划分 | 每类局部排列 [64]，切为12/26/26；加64×类别编号转成全局编号，合并得到 train/val/test 索引 [36]/[78]/[78]；同索引取 X/y | randperm(64) 每类调用一次，切片 [:12]/[12:38]/[38:]；cat 合并同用途索引；Tensor索引选行 | 三组互斥、覆盖192个原样本；输入 [36,2]/[78,2]/[78,2]，标签 [36]/[78]/[78]；两模型共用固定划分 |
| 3. 模型与评估 | 现有 A/B 两模型；建议接口 evaluate(model,X,y,loss_fn)→(loss_float,accuracy_float)，可复用为代码段 | 各模型创建前 seed123，整体to同设备，随后分别创建AdamW；评估时eval/no_grad重新前向、CE、argmax(dim=1)、布尔比较转float求mean、item取数 | 评估无backward/step，不重复创建优化器；函数返回标量而非带图Tensor；下一次更新前train |
| 4. 单组训练与记录 | 同组独立history、best_val_loss/best_epoch/best_state；400次全train更新；0及每20次评估train/val | 常规zero_grad/forward/CE/backward/step只处理train；更新后评估，历史每条可含epoch/train_loss/train_acc/val_loss/val_acc | 各21条评估记录；最低val CE与轮次/参数快照对应，epoch400指标另保留 |
| 5. 最佳快照 | 验证损失改善时，独立保存当前状态；best_val_loss可从正无穷开始，epoch0先参与选择 | state_dict返回含共享底层存储张量的字典；copy.deepcopy复制独立快照；改善时同步更新最佳损失/轮次/快照 | 不能仅赋值state_dict或model引用来冻结最佳状态；此处无须落盘/恢复优化器 |
| 6. 最终选择与测试 | 两组完成后按best_val_loss选组（相等选A），恢复该组快照，再只评估此模型的test | load_state_dict(best_state)修改现有模型，返回键匹配信息，不返回模型；最终复用评估逻辑 | 选择依据仅val；测试不进入训练/轮次/结构选择；打印两组比较及最终test指标，按实际趋势解释 |

**D05 本轮组织（2026-10-18）**：承接 D04 首轮模式设置与过拟合现象，用小张量行为实验加一次模型更新讲解 BatchNorm、LayerNorm 和 Dropout，不要求重复分类训练曲线。先定义隐藏特征归一化与随机正则化的用途，再推导归一化的均值/方差统计轴、eps、可学习gamma/beta，区别参数与running buffers。对本轮[B,C]，BN训练按列统计，LN(C)逐行沿末轴统计；LN推广到[B,T,C]时仍每个token沿C计算。说明BN默认track_running_stats=True，训练使用当前batch并更新running_mean/var/num_batches_tracked，eval使用running统计且不更新；即使no_grad，train前向仍更新buffers。BN前向和LN方差使用correction=0，BN running_var更新用correction=1；BN momentum=0.1为统计EMA权重，不是优化器动量。LN两模式均使用当前样本统计，无running均值方差。Dropout(p=0.5)训练时随机置零并将保留值乘2，eval恒等，不删除参数或断开整个计算图；no_grad不关闭随机掩码。讲清model.train/eval与梯度模式两个独立开关。

**D05 学生实现框架：`lessons/d05/16_norm_dropout.py`**。脚本开头seed42，所有浮点数据与模块统一float32/CUDA，按以下顺序手写，功能块可按段落组织，不强制函数化。

| 顺序与目的 | 输入/输出与操作 | API 参数、返回值与副作用 | 核心自查 |
|---|---|---|---|
| 1. 对比统计轴与状态 | 固定X=[[1,2,4],[2,5,6],[4,3,8],[7,8,2]]，Shape[4,3]；新建默认BN(3)/LN(3)。train+no_grad各前向；打印BN输出按dim0、LN输出按dim1的均值/方差，并记录BN前后状态 | nn.BatchNorm1d(num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)与nn.LayerNorm(normalized_shape,eps=1e-5,elementwise_affine=True)保持Shape；var(dim,correction=0)对应前向方差；running_mean/running_var/num_batches_tracked先clone后比较 | 初始gamma1/beta0，均值近0、方差近1而非严格相等；train+no_grad仍使BN计数/统计更新。列均值/方差为[3]，行均值/方差为[4] |
| 2. 验证eval及LN模式 | 切BN eval，同一X前向两次，记录输出和buffers；LN同一X比较train/eval输出；列出BN的named_parameters与named_buffers名称 | .train()/.eval()递归改变模式而非requires_grad；.named_parameters()和.named_buffers()迭代(name,tensor)；.clone()保存独立状态；allclose作浮点近似比较 | BN eval重复输出一致且buffers不变；不要求BN train/eval输出相同，也不要求BN eval输出零均值单位方差；LN同输入两模式结果一致 |
| 3. 验证随机正则化 | 单独创建全1输入U=[32,16]，Dropout(p=.5)在train+no_grad下连续两次前向，观察取值及零比例，再eval前向 | nn.Dropout(p=0.5,inplace=False)逐元素随机掩码，保持Shape；torch.ones创建输入；零比较转float取mean得零比例；no_grad不改变模块模式 | train输出0或2，零比例随机、通常约一半，不强制精确值或固定mask；两次前向间不重设seed；eval输出等于U |
| 4. 接入可训练模型 | 另建3→8 Linear→BN(8)→ReLU→Dropout(.5)→Linear→3，使用同一X与y=[0,1,2,1] long。train下CE+SGD(lr=.01)做一次更新；随后eval普通前向与eval+no_grad前向对比 | 子模块作为模型属性注册、整体to后创建优化器；CE接原始[4,3] logits和[4]标签；backward/step只在训练部分。isfinite检查梯度；clone可保留更新前参数；requires_grad观察普通输出 | 梯度存在且有限、至少参数有变化，不要求每项梯度非零或一次loss必降。eval普通前向仍可requires_grad=True，eval+no_grad输出False；同参数/input下eval输出一致，BN buffers不再改变 |

框架主线为固定输入→归一化轴/状态→Dropout随机行为→一次训练与评估，参考API说明参数/返回值/副作用与放置位置，不提供完整可运行模型或实验答案。验收核心为统计轴、BN buffers、Dropout模式和梯度模式关系及一次正确接入；无手算预测注释、下载、DataLoader、checkpoint、固定精度门槛或额外完整训练要求。可用atol约1e-5、rtol约1e-4作本例归一化统计的近似检查，eps使方差不严格等于1；本例仅检查初始gamma/beta未训练时的统计性质，不推广到任意学习后输出。

**D06 本轮组织（2026-10-21）**：以局部连接与空间权重共享说明卷积的图像建模用途，先讲单通道滑窗逐元素乘/求和/偏置，再扩展到NCHW、多输入通道求和与多个输出特征图。PyTorch Conv2d执行不翻转核的互相关，手写版本与其约定一致。明确stride是窗口起点步长、padding是边缘补零，dilation=1时输出高宽为floor((输入尺寸+2p-核尺寸)/s)+1；池化沿空间逐通道取最大/均值，不引入可学习参数，默认ceil_mode=False。区分nn.Conv2d创建注册可训练权重与F.conv2d使用传入权重直接计算，groups=1时weight=[Cout,Cin,kh,kw]、bias=[Cout]，参数量=Cout×Cin×kh×kw+Cout，空间尺寸与batch数量不增加参数。

**D06 实现框架：`lessons/d06/17_conv_pool.py`**。seed42，float32/CUDA，输入/核/偏置同设备。无下载、数据加载、分类训练或新checkpoint要求；本章只验收前向核心计算及Shape，反向实验为可选。

| 顺序与目的 | 输入/输出和实现框架 | API、参数与副作用 | 核心自查 |
|---|---|---|---|
| 1. 固定小输入 | X为1–16按行排成[4,4]，K=[[1,0],[-1,2]]为[2,2]，本轮bias=.5 | arange(start,end,dtype,device)右端不含；reshape调整Shape；tensor创建核 | 所有浮点Tensor同dtype/device；无padding、stride1时左上窗口输出8.5仅作教学示例，不要求预测注释 |
| 2. 手写单通道前向 | 建议接口conv2d_single(x,kernel,bias=0.0,stride=1,padding=0)→二维Tensor；仅单图单通道、整数对称padding/stride、dilation1。补零→计算Hout/Wout→分配输出→按输出i/j定位输入i*s/j*s窗口→逐元素乘核/求和/加偏置→保存结果 | F.pad(input,pad,mode='constant',value=0)，二维pad顺序(left,right,top,bottom)；zeros(size,dtype=x.dtype,device=x.device)准备输出；Tensor切片、乘法、sum实现窗口核心 | 核不翻转，bias每输出位置加一次；F.pad可用，F.conv2d/unfold不能代替本轮手写核心；返回正确二维Shape，不要求通用框架/边界工程 |
| 3. 对照stride/padding | 对同X/K/bias，(s,p)=(1,0)/(2,0)/(1,1)/(2,1)，输出[3,3]/[2,2]/[5,5]/[3,3] | F.conv2d(input,weight,bias,stride,padding)仅用于手写函数外对照；X加batch/channel轴为[1,1,4,4]、K变[1,1,2,2]，bias为[1]Tensor；输出选[0,0]回二维 | 每组打印Shape、最大绝对误差与allclose，使用同核/同偏置、只补一次padding，近似容差可用atol1e-5/rtol1e-4；不要求手算完整输出 |
| 4. 多通道与参数 | 随机输入[2,2,5,5]，官方Conv2d(2,3,kernel_size=3,stride=2,padding=1,bias=True)，输出[2,3,3,3] | 模块整体to设备后调用；dilation1/groups1默认，weight=[3,2,3,3]、bias=[3]；parameters/numel计算57参数 | 每个输出通道融合所有输入通道，权重跨batch/空间共享；这里是新随机可训练模块，不要求它等于前面固定K的结果 |
| 5. 池化 | 将上一步同一卷积特征分别输入MaxPool2d与AvgPool2d，kernel_size2/stride2/padding0/ceil_modeFalse，两个输出均[2,3,1,1] | 创建池化模块再传入Tensor；stride默认等于kernel_size，本轮显式2；两种池化分别接相同卷积输出，不串联 | N/C不变、只缩小H/W，pooling无学习参数，余下不足完整窗口的位置按floor规则处理；首块窗口max/mean可直接对照Tensor局部统计 |

任务意义是将公式、滑窗索引、API形状与图像特征流对应，为D07的CNN/残差网络准备。必须由学生写窗口核心并运行参考对照；仅提供接口、公式、文字与不可执行伪代码，不写完整函数。完成后按真实数值与Shape验收，不设优化速度/训练精度门槛，不要求增加多通道手写、dilation/groups分支或附加变体。

**D07 本轮组织（2026-10-22）**：把已学Conv/BN/分类/验证流程组成小型残差CNN。先讲CNN从局部特征到类别logits的数据流，再定义残差主分支R(x)、shortcut S(x)与输出ReLU(R(x)+S(x))；identity捷径适用于相同通道/stride1，通道或分辨率变化时用1×1卷积+BN投影。相加前两路Shape必须完全一致，不能依赖广播；禁止在捷径detach输入。说明加法节点的identity梯度项，最终梯度仍受后续激活等影响，不宣称梯度永不消失。BN2d对NCHW的每通道沿N/H/W统计；AdaptiveAvgPool2d((1,1))对每通道空间求均值，flatten(start_dim=1)保留batch，再接Linear输出类别分数。

**D07 实现框架：`lessons/d07/18_cnn_residual.py`**。只用本地合成数据，不下载/安装torchvision，不使用现成ResNet替代手写模块。数据准备seed42，模型创建前seed123，float32输入/long标签、实际运算统一CUDA。两种shortcut均需在同一分类器中体现；不要求普通CNN消融对照、速度优化或固定分类精度。

| 顺序与意义 | 输入/输出与实现要求 | 相关API及副作用 | 核心自查 |
|---|---|---|---|
| 1. 图像数据与固定划分 | 三类各32张[1,16,16]模板：0垂直条纹(列6:10为1)，1水平条纹(行6:10为1)，2十字(上述两处均设1、交叉处仍1)，背景0；分别加0.2标准正态独立像素噪声。每类randperm32固定分为16/8/8，train/val/test=48/24/24，保留原始索引 | zeros/tensor切片赋值构造模板，randn_like(input)返回同Shape/dtype/device噪声，full生成long标签，cat/randperm按已有方式分层；先CPU准备再统一迁移 | 输入[96,1,16,16]、标签[96]，各划分互斥且配对；噪声每样本独立，不将同一已加噪样本复制跨集合；本轮测试只反映模板噪声分布 |
| 2. 学生手写残差块 | 建议ResidualBlock(in_channels,out_channels,stride=1)。主分支Conv3×3(in→out,stride=s,pad1,biasFalse)→BN2d(out)→ReLU→Conv3×3(out→out,stride1,pad1,biasFalse)→BN2d(out)。shortcut同通道且s1时Identity，否则Conv1×1(in→out,stride=s,pad0,biasFalse)→BN2d(out)。最后两路相加再ReLU | 新子模块在__init__创建并注册，forward组织两路；nn.Identity保持输入，无学习参数；Conv1×1为每空间位置学习通道映射，stride可下采样；默认非原地ReLU | identity块8→8/s1保持[B,8,16,16]；投影块8→16/s2两路均[B,16,8,8]，确认相加的是原输入或其投影，梯度不detach |
| 3. 分类器组装 | 入口Conv1→8(3×3,s1,p1,biasFalse)→BN2d8→ReLU；块A8→8/s1；块B8→16/s2；全局平均池化→flatten→Linear16→3 | BatchNorm2d(num_features)保持NCHW、按通道统计并维护running buffers；AdaptiveAvgPool2d(output_size=(1,1))指定目标尺寸，不是核大小；torch.flatten(input,start_dim=1)合并特征轴、保留N | Shape依次[B,1,16,16]→[B,8,16,16]→[B,8,16,16]→[B,16,8,8]→[B,16,1,1]→[B,16]→[B,3]，末层原始logits |
| 4. 小规模训练与评估 | 模型统一CUDA后AdamW(lr=.01,weight_decay=0)，CE，train batch16/shuffleTrue/num_workers0，每epoch3次更新，共20epoch。首个真实训练batch在backward后观察主分支和投影参数梯度及一次更新；不额外要求完整梯度测试 | 可复用学生Dataset；可选TensorDataset(X,y)逐样本返回元组，输入首轴必须相同；DataLoader组织批次。每epoch开头train，只有train执行backward/step；评估直接复用evaluate(model,X,y,loss_fn)→float loss/acc | 梯度存在有限、至少参数变化，不能要求全部非零。训练前epoch0及每epoch结束eval/no_grad重算完整train/val CE/accuracy，记录21点，日志每5epoch即可 |
| 5. 最佳状态和最终测试 | 按已评估轮次最低val CE保存best_epoch/best_val_loss及deepcopy(state_dict)，包含BN buffers；20epoch结束恢复最佳状态，再仅一次评估test | state_dict含参数和running统计，deepcopy独立快照；load_state_dict载入现有模型后eval/no_grad；不要求落盘/恢复优化器或训练调度 | 验证不更新BN，测试不参与调参；报告初始/最终train-val、最佳epoch/val和test结果，不能由同模板加噪数据推断真实图像泛化 |

验收聚焦图像NCHW数据、两种shortcut及逐元素残差相加、全局池化分类头、正确梯度/训练与验证选择；无固定准确率/损失门槛、完整ResNet家族或额外数据集项目要求。学生根据自己的模块结构解释一处identity和一处projection为什么能相加即可，接受聊天回答，不强制预测注释；代码已清楚展示正确数据流时不重复设置问答关卡。

**D08 本轮组织（2026-10-22）**：承接D07已实现的模型状态落盘，区分恢复预测与恢复同一训练过程。说明AdamW的step/一阶和二阶矩、BN参数之外的running buffers、已完成epoch/global_step、随机序列当前位置与DataLoader打乱状态、配置与指标历史。用同环境的“连续4轮”与“2轮保存退出、重新启动再训练2轮”对照，目标是正确接续训练，不设分类精度或逐位相等门槛；不是P-D整体验收。最佳验证模型和当前训练状态用途不同，本实验在epoch边界保存当前状态，不先恢复D07最佳模型。

**D08 实现框架：`lessons/d08/`（入口 `train.py`）**。2026-10-23 学生已采用 config/data/model/engine/checkpoint/compare/train 多模块草稿；产物统一放在 `lessons/d08/outputs/`。原单文件建议已被此结构取代，模块拆分不增加验收要求。复用学生TinyResNet/ResidualBlock和D07数据逻辑，不重写新模型；旧脚本顶层会执行训练，不要求直接import旧脚本，可自行复用类定义。三次独立进程可用mode变量区分baseline/interrupt/resume，命令行参数和函数拆分为可选组织。共同设置模型seed123、loader专用CPU Generator seed456、batch16/shuffleTrue/num_workers0，AdamW(lr=.01,weight_decay=0)，无scheduler/AMP/多进程。cuDNN benchmark=False、deterministic=True以减少卷积非确定性；仅同设备/版本作数值接近对照，不承诺跨设备或版本完全一致。

| 顺序与目的 | 输入/输出与操作 | API、参数/返回/副作用及位置 | 核心自查 |
|---|---|---|---|
| 1. 固定数据和实验起点 | baseline按D07噪声图像/固定划分生成一次，将CPU train图像[48,1,16,16]/long标签[48]、val图像[24,1,16,16]/标签[24]和train样本ID保存到d08_data.pt；interrupt/resume读取同一文件。模型初始化种子放在数据准备之后 | torch.save(obj,path)立即写盘返回None；torch.load(path,map_location='cpu',weights_only=True)返回原字典并把Tensor映射CPU。TensorDataset(X,y,ids)三者首轴相同，ids只记录不喂模型；batch图像/标签toCUDA，ID留CPU | 三路使用同一批实际样本及配对、模型结构/初值/配置相同；不要用D07训练后权重作为起点 |
| 2. 一轮训练与日志 | 每轮3次真实step；记录每批ID列表、global_step，以及epoch结束eval/no_grad全量train/val CE和accuracy的Python数值。history每轮一行，字段epoch/global_step/train_loss/train_acc/val_loss/val_acc | torch.Generator(device='cpu')创建独立生成器，manual_seed(456)改变其起点并返回自身，DataLoader(generator=g)让shuffle和迭代器种子使用它；tolist记录CPU ID，item取指标数值；每轮开头model.train | baseline4轮12step；interrupt2轮6step；恢复只追加第3、4轮，到12step/history4行，评估不更新BN |
| 3. 完整epoch边界快照 | 字段model_state、optimizer_state、completed_epochs、global_step、torch_rng_state、cuda_rng_state_all、loader_rng_state、history、batch_order、config。config仅基本数值/字符串，记录模型/种子/批量/lr/wd/目标轮数等；随机状态在本轮评估和记录后、下一迭代器之前采集 | model.state_dict含参数/BN buffers但不含train/eval标志与梯度；optimizer.state_dict含state/param_groups；torch.get_rng_state→CPU ByteTensor，torch.cuda.get_rng_state_all→CPU ByteTensor列表，g.get_state→CPU ByteTensor。取状态不推进随机序列；立即save可不deepcopy，留内存继续训练后再保存则需独立快照 | interrupt保存当前第2轮状态，不恢复best；同一时点的模型/优化器/RNG/计数配套，未引入Python random/NumPy无需额外RNG |
| 4. 新进程恢复 | 先读数据/配置/检查点，创建模型并整体toCUDA，创建绑定该模型的AdamW与loader；载入模型和优化器、恢复计数/历史；最后恢复CPU/CUDA/loader RNG，再进入下一轮 | model.load_state_dict(state,strict=True)默认原地复制返回缺失/多余key信息；optimizer.load_state_dict(state)原地恢复返回None，含参数组配置，当前默认AdamW会将moment匹配参数设备，step可能保留CPU；torch.set_rng_state、torch.cuda.set_rng_state_all、g.set_state恢复状态并返回None。载入CPU映射保证ByteTensor状态不被误搬CUDA | 不把load_state_dict返回值赋给model/optimizer；不在恢复RNG后重新seed或提前迭代loader；completed_epochs=2表示第3轮开始，下一更新前清grad并train |
| 5. 三路对照和可读日志 | baseline4轮输出d08_baseline.pt/json；interrupt2轮输出d08_resume.pt；新进程resume输出d08_resumed.pt/json并与baseline比较。json记录4行history，resume保留前2行再追加 | json.dump(history,fp,indent=2,ensure_ascii=False)将基本类型写入打开的UTF-8文本文件，返回None，不直接序列化Tensor。torch.allclose(a,b,rtol=1e-4,atol=1e-5)返回bool，比较前同设备；整数buffers用equal，浮点参数/BN buffers按容差，并报告最大参数绝对差 | 两路batch_order全部12批应一致，完成轮数4/step12；全部模型状态数值接近，优化器恢复含step/moments，后两轮验证指标接近。比较失败先定位状态或数据，不用提高精度门槛代替诊断 |

主流程伪代码为“准备固定数据 → 创建同起点模型/优化器/独立loader生成器 → 若resume则恢复全部状态且最后恢复RNG → 按剩余轮数训练/评估/追加记录 → 保存当前完整状态与日志 → resume与baseline对照”。第一步只需复用已有类，完成固定数据文件与一轮训练记录，再扩展快照/恢复；不要求先实现完整工程框架。不要求中途batch精确恢复、分布式、额外消融、绘图、安装日志库或重复测试集评估。实际学生进度只记PROGRESS.md。

**D09 当前组织（2026-10-26）**：学生授权在Windows CPU环境先学习下一节；D08入口确定性配置已审阅正确，CUDA最终复核继续保留，不据此标记D08通过。D09先集中讲解/实践梯度累积与全局梯度范数裁剪，说明AMP自动选择运算精度、FP16梯度缩放以及先unscale后clip的配合顺序。CUDA AMP、显存/速度/数值对照待CUDA环境，当前CPU子任务通过不等于整章AMP实验已完成。不改D08已正确的恢复逻辑或追加长训练。

**D09 CPU实现框架：`lessons/d09/20_grad_accum_clip.py`（学生手写）**。本轮使用熟悉的线性回归隔离更新机制，避免TinyResNet的BatchNorm批统计使微批与大批本来就不等价。CPU float32，X=[[-2],[-1],[1],[2]]、Y=2X+1，均[4,1]；seed42，三个独立nn.Linear(1,1)先复制同一初始state_dict，再创建各自SGD(lr=.1,momentum=0,weight_decay=0)，无Dropout/BN。只做每组一次参数更新，不要求多轮收敛、下载、保存、计时或精度门槛。

| 功能块/目的 | 输入输出与操作 | API及使用位置/副作用 | 核心自查 |
|---|---|---|---|
| 1. 准备同起点 | X/Y为CPU float32 [4,1]；三模型weight [1,1]/bias [1]；输出三组独立模型/优化器 | torch.manual_seed在初始化前；nn.Linear(1,1)输出[B,1]；load_state_dict在任何更新前复制初值，不将返回报告赋给模型；SGD绑定各自参数 | 三组初值相同、参数对象独立；函数化/多模块为可选 |
| 2. 完整batch参考 | 四样本MSE mean标量，清grad→前向→backward→记录梯度→step一次 | nn.MSELoss默认对全部误差元素mean；backward累积grad不更新参数；zero_grad(set_to_none=True)清梯度，step更新参数 | backward_calls=1、optimizer_steps=1；保留梯度和更新后参数供比较 |
| 3. 两个微批累积 | 按原顺序分两个[2,1]微批；每个mean loss除2后立即backward，两个微批结束后step一次 | 清grad仅在窗口开头；microbatch循环内不step/zero_grad，不需retain_graph；若梯度记录后还要修改，可detach().clone保存 | backward_calls=2、optimizer_steps=1；与参考组梯度及更新后参数allclose(rtol1e-5,atol1e-6)；解释不等长微批应按样本数加权，但不布置额外变体 |
| 4. 累积后裁剪 | 第三组重复相同累积，step前将全部参数梯度合并视为一个向量，L2阈值0.1，输出前后范数及更新后参数 | nn.utils.clip_grad_norm_(model.parameters(),max_norm=.1,norm_type=2)在所有backward之后且step之前调用，原地改grad，返回裁剪前范数；裁剪后范数由各grad平方求和再开根计算 | 裁剪后总范数≤阈值（允许小浮点误差），每组仍仅step一次；不要求裁剪组与未裁剪组参数一致 |

主流程伪代码：“准备四样本与三个同起点模型 → 完整batch参考一步 → 两微批按1/2缩放累积并更新一步 → 同样累积后裁剪再更新一步 → 对照梯度/参数/更新次数与裁剪前后范数”。记录可用打印或字典，必要内容为方法名、backward次数、step次数、参考/累积梯度与更新后参数、裁剪前后范数，不强制记录文件或函数签名。当前第一步只准备数据与同起点模型，再完成参考组。AMP后续使用autocast选择合适精度，CUDA FP16场景GradScaler负责scale/backward、窗口结束unscale、clip、scaler.step/update；不在当前CPU脚本中直接运行CUDA API，不强制手工model.half。

**D09 CUDA AMP组织（2026-10-29）**：恢复到已具备CUDA的Linux设备，先依据当前学生文件核验已有CPU累积/裁剪，不要求重复编写。下一练习建议 `lessons/d09/21_amp_accum_clip.py`，复用20_grad_accum_clip.py的数据、同起点模型与全局范数计算，比较CUDA FP32与CUDA FP16 AMP各一个有效batch更新。CPU任务与AMP各自按实际证据记录，D08最终CUDA复核不混入本次教学。作者资源选nanoGPT train.py约274–297行的autocast/累积/缩放/裁剪/更新路径；旧AMP写法用本地Torch2.10的torch.autocast和torch.amp.GradScaler校对，不复制完整训练循环。

**AMP原理和范围**：混合精度按运算选择dtype，不手工model.half；本轮参数/输入/目标为CUDA float32，autocast中的Linear输出float16、MSE标量float32，FP32参数的grad仍float32。FP16中间梯度可能下溢，GradScaler将损失乘S使反向梯度放大，累积完成后除S还原；累积窗口内S不变，且只unscale一次，clip在unscale之后。非有限梯度时scaler.step可跳过更新，返回None不能作为是否更新的判据。BF16指数范围较大，通常不需FP16式缩放；本机支持BF16不改变本次FP16机制练习，不强制额外BF16组。AMP不保证总显存减半或任何小模型更快，性能对照延后到有代表性的模型按需观察，不作本轮门槛。

| 功能块/目的 | 输入输出与操作 | API参数、返回、副作用与位置 | 自查 |
|---|---|---|---|
| 1. 同起点与FP32参考 | 沿用X/Y=[4,1]、两个独立Linear1→1、seed42、CUDA float32参数与数据，各自SGD lr.1/无动量衰减。参考组两个[2,1]微批，每个mean MSE除2累积，clip阈值.1后一次step | 复用已通过的累积/裁剪逻辑，不引入BN/Dropout；模型整体toCUDA后绑定优化器，复制初值在任何更新前完成 | 两组初值相同、参数独立，FP32组也裁剪；只一个有效batch，不比较收敛 |
| 2. autocast前向 | 每微批在上下文内做Linear、MSE和loss/2；输出预测[2,1]及标量loss，反向在上下文外 | torch.autocast(device_type='cuda',dtype=torch.float16)返回上下文管理器，进入/退出控制范围内eligible运算精度，不永久改模型/输入。with只包前向和loss | 记录参数/输入/预测/loss dtype，不把所有Tensor变half或把loss必须half当目标 |
| 3. 缩放后累积 | AMP模型只创建一个scaler，清grad一次；两个微批分别缩放归一化后的loss并backward | torch.amp.GradScaler('cuda',init_scale=128.0)返回有状态scaler，放在微批循环外；scale(loss)返回新缩放Tensor，不原地改loss。get_scale返回Python float；init_scale128是本小实验配置，避免默认大scale在第一次小窗口引发溢出跳步，不是通用最佳值 | 仍是loss/2，不以scale替代平均；微批中不step/unscale/update，不需retain_graph |
| 4. 还原/裁剪/更新 | 两次backward完毕，观察放大梯度范数；还原后统一clip .1，重算裁剪后范数，再更新一次 | scaler.unscale_(optimizer)原地除grad并检查非有限，返回None，每窗口一次；clip_grad_norm_返回裁剪前的真实范数；scaler.step(optimizer)检查后调用/可能跳过optimizer.step，返回底层step结果；scaler.update()窗口末调整scale返回None | 不能裁剪放大的梯度、不能额外optimizer.step；记录scale前后、还原前后范数和参数实际变化，默认growth_interval2000，本次成功更新后scale不变是正常的 |
| 5. 有限对照与源码映射 | 两组更新后参数有限且AMP实际更新，裁剪后范数≤.1（容许小误差），参数差异报告而非要求完全相同；打印/字典即可。学生可再对照作者该段代码认出操作顺序 | 复用detach().clone、isfinite、grad_norm；可用allclose(rtol1e-2,atol1e-3)作本小实验的接近参考，不作为所有模型的普遍误差标准；scaled/unscaled范数比应接近本窗口S | 不要求AMP与FP32逐位相等、固定速度/内存优势、故意制造溢出、长期训练、绘图或新工程 |

主流程伪代码：“准备同起点的两组CUDA FP32模型 → 复用普通累积/裁剪参考一步 → AMP窗口开头清grad → 每微批autocast前向/MSE除2，退出上下文后scale/backward → 全部微批结束unscale → clip → scaler.step → scaler.update → 对照dtype、范数、scale及参数”。第一步是迁移数据/两个同起点模型并跑通FP32参考；核心AMP窗口由学生手写，函数化/模块化/计时/显存观察可选。AMP验收后再开展D10的必要验证并进入语言模型主题。

**D10 本轮组织（2026-11-03）**：以学生D01曾遇到的预测[4,1]、目标[4]导致MSE错误广播为真实动机，讲解测试是验证可观察的性质，区分shape相等、有限梯度、真实参数更新及单批过拟合各能证明什么。复用D09四样本线性回归，学生自行写小型检查脚本 `lessons/d10/22_training_checks.py`；CPU float32，输入X=[[-2],[-1],[1],[2]]、Y=2X+1，seed42，Linear(1,1)、SGD(lr=.1,momentum0,weight_decay0)，固定同一batch约40次更新。只验证当前模型和指定输入，无需下载、pytest、训练大型模型或重写分类器。正确性断言与预期失败以Python assert/try-except表达；用提示而非完整代码。2026-11-05学生明确将本次验收限定为Shape守卫、weight梯度及SGD更新，不单独检查bias或比较初末loss；实际提交 `lessons/d10/main.py`，该范围通过证据见PROGRESS.md Session097，不把未执行的过拟合对照记录为已验证。

| 功能块/目的 | 输入输出与操作 | API参数、返回、副作用/位置 | 核心自查 |
|---|---|---|---|
| 1. 固定输入和防广播形状检查 | X/Y float32 CPU [4,1]，模型输出prediction [4,1]；定义或局部实现形状检查，检查通过返回None；将Y沿末轴squeeze得到bad target [4]，交给同一检查，预期触发AssertionError并捕获 | manual_seed(42)设初始化；nn.Linear(1,1)返回[B,1]；Tensor.squeeze(dim=-1)仅长度1时移除指定轴，返回Tensor不改变原Y；assert prediction.shape==target.shape在MSE之前检查；try/except AssertionError专门捕获预期异常，未发生异常要使测试失败 | 正确目标通过，错误[4]必须被守卫拒绝，别将预期失败误报程序崩溃；不计算错误target的MSE |
| 2. 一次反向的梯度检查 | 零梯度后计算标量MSE并backward，对weight[1,1]/bias[1]的grad检查存在、shape一致且有限；记录独立快照 | optimizer.zero_grad(set_to_none=True)清grad；loss.backward()累积梯度；torch.isfinite(tensor).all()返回标量bool Tensor；detach().clone()保存数值快照，不保留计算图或共享存储 | 每个参数的grad不为None、shape与参数相同、有限；至少一个非零梯度（具体本批seed/模型），不要求所有情况所有梯度均非零 |
| 3. 检查实际SGD更新 | 快照参数p及其梯度g，调用一次SGD.step后，比较各参数与p−0.1g，至少一个参数改变；本题SGD无momentum/weight_decay | optimizer.step()原地更新参数、SGD返回None；torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)通过返回None，失败抛AssertionError，或torch.allclose+assert | 参数梯度用于真正的优化器；不把p变化当作梯度本身一定正确的证明，也不将一般AdamW更新误写成SGD公式 |
| 4. 单批过拟合小实验 | 在同一个固定X/Y上总计40次SGD更新（含上一步首次更新）；记录初始loss和第40步后用当前参数重新计算的loss，报告下降及可选参数 | 每步zero_grad→forward→检查shape→MSE→backward→step；结束时no_grad重新前向/loss，item返回Python数值；原地step只更新参数，不会更新先前的loss对象 | 有限且最终loss相对初值明显下降；这只说明模型能拟合该批，不能证明泛化，也不能单凭下降证明样本配对或完整算法无误 |

主流程伪代码：“固定小数据/模型→校验正确shape、构造错误target并验证守卫捕获→正确标签做一次反向、检查梯度→快照并检查一次真实SGD更新→原batch再训练剩余步→重算最终loss”。学生可以直接编写本课单个脚本，或自行复用D09已验证的模型/数据片段；因旧脚本顶层有初始化，不要求直接import或强制函数化。不强制pytest、额外边界集、数值梯度或指定loss阈值；有效断言应能捕获已知广播问题和错误更新。第一步是准备数据和shape守卫；完成后集中验收。来源：PyTorch官方broadcasting语义，以及nanoGPT作者train.py的get_batch输入/错位目标路径（D10仅联系测试动机，不提前要求实现语言模型）。

**验收项目 P-D（2026-10-28调整）**：综合核对已有D阶段学生实现，确认数据划分、模型/损失、梯度更新、训练/评估模式、日志与检查点的核心行为正确；优先复用D07/D08及D10的实际证据，不要求另写完整分类项目、指定模块/测试文件数量或新增README。必要检查针对已遇到的真实Bug和核心性质；已有有效检查可复用。CUDA待补范围单独记录，不把静态检查冒充实测，也不阻塞不依赖CUDA的E阶段主题。学生仍自己实现模型与训练核心，无SOTA/固定精度要求。

## 8. 阶段 E：Transformer、语言建模与 mini-GPT（E01–E14）

**目标**：手写最小可训练的 Decoder-only GPT，并理解与主流 Transformer 变体的关系。这是首个大型里程碑。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| E01 | 序列建模、token、词表、字符级 tokenization | 实现字符编码和解码 |
| E02 | n-gram、语言模型概率分解、next-token 目标 | 从原始字符串构造输入/标签 |
| E03 | Embedding与上下文表示；位置表示按GPT需要引入 | 手写小型上下文预测模型，追踪 `(B,T,C)` |
| E04 | Attention 的 Q/K/V 与缩放点积 | 实现分数/加权汇总并核对Shape，不强制整张量手算 |
| E05 | Causal mask、softmax、数值稳定性 | 验证未来 token 不可见 |
| E06 | Single-head Self-Attention 独立实现 | 写模块与维度测试 |
| E07 | Multi-head Attention、head reshape/transpose | 验证多头拼接和投影 |
| E08 | Residual、LayerNorm、MLP、Pre-Norm | 手写 Transformer Block |
| E09 | Decoder-only GPT 堆叠、权重共享/初始化 | 实现模型并计算参数量 |
| E10 | 交叉熵、Teacher Forcing、训练小 GPT | 证明模型可过拟合少量文本 |
| E11 | 自回归采样、temperature/top-k；top-p可选 | 实现文本生成器 |
| E12 | checkpoint、验证困惑度、实验记录 | 训练/推理分离并复现输出 |
| E13 | 阅读 nanoGPT 并建立论文/模块映射 | 输出数据流、调用图、参数表 |
| E14 | 复杂度、KV Cache、SDPA/FlashAttention、推理效率（首个mini-GPT之后扩展） | 与手写 Attention 做一致性对照 |

**实际授课按完整主题组合，保留E01–E14索引以追溯，不机械逐API开课：**

| 主题课块 | 对应索引 | 原理与学生代码产物 | 核心验证及参考位置 |
|---|---|---|---|
| 1. 从文本到可采样的语言模型 | E01–E02 | 字符词表/encode-decode、next-token概率分解、错位输入标签与bigram基线；IDs [B,T] long，logits [B,T,V] float | 编解码往返、标签右移一位、loss有意义及采样ID合法；Zero to Hero makemore bigram |
| 2. 从查表到学习上下文 | E03 | Embedding、固定长度上下文和小MLP预测器；[B,T]→[B,T,C]→类别logits，联系已学CE/训练；GPT用的learned position在需要时加入 | 参数收到梯度、输出与目标对应、正确训练/验证划分；makemore MLP，nanoGPT token/position embeddings |
| 3. 一个完整的因果注意力头 | E04–E06 | Q/K/V、缩放点积、softmax轴、causal mask与加权汇总一并实现；[B,T,C]→分数[B,T,T]→输出[B,T,H] | 概率沿key轴归一、改变未来token不影响过去输出、梯度有效；GPT课程及CausalSelfAttention |
| 4. 一个可堆叠的Transformer Block | E07–E08 | 多头拆分/合并、输出投影、Pre-LN、残差与MLP组成完整Block；[B,T,C]→[B,T,C] | 两路残差Shape相同、head维度正确、无未来泄漏；nanoGPT Block/MLP/LayerNorm |
| 5. 训练自己的小GPT | E09–E10 | token/position表示、Block堆叠、分类头与序列CE接入；logits[B,T,V]与targets[B,T] | 可拟合小文本，训练/验证分开，报告损失与资源配置；nanoGPT GPT.forward与train |
| 6. 让模型生成并保存成果 | E11–E12 | temperature/top-k、自回归上下文裁剪、生成、恢复推理与验证困惑度；复用已学checkpoint能力 | ID范围、输出长度、恢复后预测一致，不要求生成语义质量或固定perplexity；GPT.generate/sample.py |
| 7. 对照nanoGPT并解释差异 | E13 | 将本人实现对应data/get_batch、model/Block、train/optimizer、generate/sample，选一处小型数值/行为对照 | 说清公式/Shape/数据流与不同设计；不抄完整答案，不先做DDP或大规模GPT-2复现 |

每块可跨数次会话。上表是课程目标和产物框架，不是一次性作业；开课时仍按学生当前代码给出目的→输入输出→操作/API→自查→第一步。源码阅读课应有一个小型代码观察/对照，纯概念内容也配套可计算产物；不为已通过的相同知识重复建项目。E14、完整BPE及现代结构扩展在首个P-E闭环之后安排。

**E01–E02 本轮组织（2026-11-05）**：用一个本地极小语料贯穿字符词表、编码/解码、错位next-token监督、可训练bigram查表、交叉熵和逐字符采样。先说明token/词表/ID、语言模型条件概率与链式分解；bigram只依赖当前字符，输出是下一字符分布，没有注意力/长上下文；因此它是后续GPT的简化基线而非GPT本身。一个token在本实验为一个Python字符（包含空格和换行），不要求BPE或外部数据。建议学生手写 `lessons/e01/main.py`，CPU足够；`lessons/e01/outputs/`按课程约定用于可选产物。自有固定字符串 `"hello world\n"` 重复8次，长度96、字符词表V=9；`sorted(set(text))`确定ID次序，词典char→ID及反向表需一致。模型种子42、AdamW lr=.05/weight_decay0，训练整段序列100次小更新；无需固定loss门槛或文本语义质量，不强制初末loss比较。

| 功能块/目的 | 输入输出与操作 | API参数/返回值/副作用及位置 | 核心自查 |
|---|---|---|---|
| 1. 字符词表/往返 | 固定text长度96、V=9；构造字符与ID一一对应，encode(str)→list[int]、decode(ids)→str；CPU普通Python对象 | sorted(set(text))去重并排序返回列表；enumerate产生(索引,字符)对；字典索引查ID、列表索引查字符，join连接字符；字符串不参与梯度 | 原文decode(encode(text))精确相等；换行和空格均入词表；训练/生成共用同一映射 |
| 2. 错位数据 | 全文IDs long CPU [96]；取前95与后95，增加batch维，x/y均[1,95]、torch.long、同设备 | torch.tensor(ids,dtype=torch.long)创建ID张量；[:-1]/[1:]分别取相邻输入/标签，unsqueeze(0)插入batch轴并返回新Tensor/view；ID无梯度 | x每个位置对应原文当前位置，y为紧跟的下个字符；x[:,1:]与y[:,:-1]一致；不要只看shape |
| 3. bigram查表与CE | nn.Embedding(V,V)可学习weight[V,V]；x[1,95] long→logits[1,95,V] float；每位置仅按当前ID取行，该行每列是下个字符得分 | nn.Embedding(num_embeddings=V,embedding_dim=V)注册可训练矩阵，输入ID范围[0,V)；CrossEntropyLoss输入raw logits.reshape(-1,V) [95,V]、targets.reshape(-1) [95] long，返回标量mean；不用提前softmax | 对齐与flatten次序一致；logits维度和标签dtype正确，loss有限；此用法将表行直接解释为分类logits，一般Embedding(V,C)可输出普通特征 |
| 4. 训练小基线 | seed42、AdamW(model.parameters(),lr=.05,weight_decay=0)，整段序列100次更新；long x/y与FP32参数同CPU | 每次zero_grad→embedding→CE→backward→step，记录可打印loss.item()，不用记录或比较初末loss；反向写参数grad，step原地修改参数 | 至少一次真实参数更新、loss/梯度有限；不要求曲线、特定值/准确率或泛化结论 |
| 5. 自回归采样 | 初始'h'已在词表，将其编码为[1,1] long；逐次只送最近一个ID，查表得[1,1,V]，取最后位置[1,V]，softmax后采样[1,1]，与已有序列沿dim1拼接；循环20次后总长21，再decode | no_grad用于采样；torch.softmax(logits,dim=-1)返回概率、按词表维归一；torch.multinomial(probs,num_samples=1)返回long索引且随机采样；torch.cat(...,dim=1)返回拼接新Tensor；新ID作为下一轮输入 | ID范围[0,V)，输出长度21且前缀h，所有字符均在词表；随机生成不要求固定句子或语义；由于模型只看最后一个字符，无法记忆更长语境 |

主流程伪代码：“固定文本→建立词表/编码并验证往返→错位构造x/y→创建[V,V]转移得分表→CE训练→从h逐次采样20个ID→解码”。首先实现词表/encode/decode并完成往返，再逐块接数据/模型/训练/生成；函数化与日志文件可选，训练与生成核心由学生编写。仅验证此小文本训练数据，不声称验证集/泛化或成熟GPT；D08最终CUDA复核独立保留。参考作者makemore的字符级bigram主题及PyTorch2.10 Embedding/CE/multinomial API，不复制参考整段实现。

**E03 本轮组织（2026-11-10）**：承接已验收bigram，学习Embedding作为可训练字符表示、固定3字符上下文与MLP next-token预测。输入[ B,3 ] long→Embedding(9,4)特征[ B,3,4 ]→保序flatten[ B,12 ]→Linear12→32/Tanh→Linear32→9 logits[ B,9 ]，目标[ B ] long；CE以原始logits和每个上下文之后的一个字符为目标。上节[V,V]直接是候选字符分数，本节[V,4]是学到的特征，需通过MLP输出9类分数；相同字符共享Embedding行，梯度更新这些向量。固定位置拼接使不同槽位可被后续权重区分，但不能保证任意初值对换序均有不同输出；并非GPT的位置编码或注意力。

建议学生手写 `lessons/e03/main.py`，复用当前 `lessons/e01/main.py` 的固定文本/词表思路，不要求导入含顶层对象创建和打印的旧脚本。固定原文 `"hello world\n"`×8（96字符），按原字符位置先拆train前72和val后24，各自独立滑窗，不跨分界；只用训练段构造排序词表，9字符均在训练段出现。窗口T3，train 72−3=69、val 24−3=21；输入x_train[69,3]/y_train[69]、x_val[21,3]/y_val[21]均CPU long，模型float32 CPU。样本索引s的输入为segment[s:s+3]，目标segment[s+3]，范围s=0..len(segment)−4。相邻窗口高度重叠；此数据重复结构使train/val上下文也会相同，字符位置不交叉不等于独立样本或真实泛化测试。

| 功能块/目的 | 输入/输出与操作 | API参数、返回、副作用和使用位置 | 核心自查 |
|---|---|---|---|
| 1. 划分/词表 | text[96]先按位置分train[72]/val[24]；字符映射由train构造，V=9，编码两个分段；每段ID序列long | Python切片不改原文本；sorted(set(train_text))返回字符表，dict查ID；torch.tensor(...,dtype=torch.long)返回CPU索引张量 | 两段不共享原文位置，训练词表覆盖val字符；同文本重复结构不宣传独立泛化 |
| 2. 有序上下文/标签 | 对各段分别滑动窗口，输入[N,3]、目标[N]；train N69、val N21 | 每个s切3个有序ID，标签取紧接第4个ID；可torch.stack(xs)→[N,3]、torch.stack(ys)→[N]，或由列表建tensor，保持long | 首几项`hel`→`l`、`ell`→`o`、`llo`→空格；最后目标不越界，train/val窗口不跨分界 |
| 3. Embedding+MLP | 模型输入[B,3] long；中间[B,3,4] float32、flatten[B,12]、隐藏[B,32]；输出[B,9]，每条上下文预测一个ID | nn.Embedding(num_embeddings=9,embedding_dim=4)注册weight[9,4]；flatten(start_dim=1)保留batch及位置顺序；nn.Linear(12,32)、nn.Tanh()、nn.Linear(32,9)在__init__注册并在forward调用，返回raw logits | B=69时[69,3]→[69,3,4]→[69,12]→[69,32]→[69,9]，不是[69,3,9]；不先softmax再CE |
| 4. 训练/验证 | seed42，AdamW(lr=.03,weight_decay=0)，全train batch进行200次小更新；CE输入logits[69,9]、y[69] long，同CPU | CrossEntropyLoss mean返回标量；每次zero_grad/forward/loss/backward/step；Embedding和Linear参数应有有限梯度且至少参数更新。最后eval/no_grad对val做一次独立前向和CE | val只参与评估，不参与backward/step/选超参；无需初末loss对比、固定精度或泛化结论 |
| 5. 相同末字符的不同上下文 | 比较两个真实train窗口hel与ell，二者均以l结尾，但真实下一字符分别为l、o。输入[2,3] long→logits[2,9]→probabilities[2,9] | model.eval()管理模型行为；no_grad关闭评估梯度记录；softmax(dim=-1)沿9类转概率；展示两个上下文实际目标及其预测分布或候选概率 | 不强制argmax/概率达到预设数值；形状与两个目标配对正确，观察多字符信息如何进入MLP，不能仅凭初始随机输出不同就断言模型已学会泛化 |

主流程伪代码：“文本按字符位置分段→训练段建词表并编码两段→各自滑窗/目标→Embedding与MLP前向→train 200次更新→独立val评估→观察hel/ell预测”。第一步为分段、词表和首个窗口；学生手写核心，函数化及连续采样是可选，固定额外测试数、预设准确率、下载和独立项目不要求。作者资源仅对应makemore.py的MLP类“多个token向量拼接预测”的思想，数据边界/训练协议以本课定义为准，官方2.10 Embedding和Linear API为接口来源。

**E04–E06 本轮组织（2026-11-11）**：将Q/K/V、缩放点积、causal mask、softmax和Value汇总合成一个完整的单头自注意力主题，服务下一步手写Transformer，避免将各API拆为零碎章节。通过学生自己实现的一个模块观察“每个当前位置只能从自身及过去位置聚合内容”，不要求同时构建完整GPT/Tokenizer/训练工程。先定义行是query位置i、列是key/value位置j，score_ij=(q_i·k_j)/sqrt(d)，j>i的分数屏蔽，再对j轴softmax，输出o_i=Σ_j a_ij v_j。缩放用于控制点积尺度；未来位置屏蔽在softmax之前，保留对角线，防止某行全部−inf。用零可见分数的例子展示T=4时行权重[1,0,0,0]、[1/2,1/2,0,0]等；随机模型的实际权重无需均匀。因果性不等同于位置编码，后者待后续模型主题。

**作业：`lessons/e04/main.py`**。学生手写SingleHead或等效nn.Module：CPU float32输入X[2,4,8]；三独立nn.Linear(8,4,bias=False)生成Q/K/V各[2,4,4]；手写矩阵计算、因果遮罩、权重和输出。forward可返回(output,weights)，shape分别[2,4,4]/[2,4,4]；无需训练分类器、下载数据、多头或直接用现成SDPA替代核心。测试随机输入seed42，基于现有autograd知识校验梯度，不设速度/概率/模型准确率门槛。

| 顺序/目的 | 输入输出与操作 | API参数、返回、副作用和使用位置 | 自查 |
|---|---|---|---|
| 1. Q/K/V投影 | X [B,T,C]=[2,4,8] float32 CPU，Q/K/V均[B,T,d]=[2,4,4] | nn.Linear(in_features=8,out_features=4,bias=False)三次分别注册独立参数；forward对同一X分别调用，Linear在末轴变换，保留B/T | 三个矩阵独立并参与计算图，B/T不丢；输入可requires_grad=True做反向检查 |
| 2. 分数与因果遮罩 | Q@K转置后[2,4,4]，按sqrt(4)缩放；mask[T,T] bool/CPU，允许j≤i | K.transpose(-2,-1)返回交换末两轴视图，@按batch矩阵乘；math.sqrt(d)返回数值；torch.ones((T,T),dtype=bool,device=x.device).tril()是允许位置示意；scores.masked_fill(禁止条件,-inf)返回新Tensor，不原地改原score，mask可广播到batch | 行i查询、列j被查询；未来j>i屏蔽，对角线和过去允许；每行至少一项可见，mask反向会屏蔽语义错误 |
| 3. 权重和加权值 | softmax按最后一维key位置得到A[2,4,4]；A@V→out[2,4,4] float32 | torch.softmax(masked_scores,dim=-1)返回同shape概率；@与V的时间轴对齐；forward返回out及weights供验证 | 每行和接近1，上三角未来位置为0，输出finite；它是每个query对可见value的加权和，非概率表直接等于输出 |
| 4. 因果性和梯度验证 | 同一个模块和输入，复制X，改变最后一个token的向量，再做第二次前向；比较位置0..2的输出；对原输出标量loss反向 | detach().clone()独立副本，在no_grad下生成对照输入；torch.allclose按容差返回bool；可mean(out.square())作为标量，backward累积梯度；检查三投影参数grad存在有限 | 前三个位置不受未来token变化影响、权重上三角为0、概率行归一；无需要求最后一个输出必改变或全部梯度非零；核对结果而非形状/打印alone |

主流程伪代码：“构造X及单头模块→三路投影→分数/缩放→上三角屏蔽→softmax→乘V→权重/因果与反向检查”。第一步先在模块中创建Q/K/V并确认三个输出shape，再补后续功能块。允许学生用函数组织、输出完整权重或打印检查结果；无需固定种子精确数值、整矩阵手算、额外模型/多头/性能/整套训练，也不假设位置编码已完成。参考Attention Is All You Need缩放点积与nanoGPT CausalSelfAttention思想，教学实现先自己写核心后再看作者源码；官方torch.masked_fill和softmax用于接口校对。

**E07–E08 本轮组织（2026-11-11）**：以一个可堆叠的因果Transformer Block为完整主题，复用E04–E06学生的SingleHead，不再重写单头。输入X [B,T,C]=[2,4,8] float32 CPU，两个独立头各接收全部C特征并投影至d=4，分别输出[B,T,d]与权重[B,T,T]；沿特征维拼接并Linear(C,C)输出[B,T,C]。Pre-LN结构 u=X+MultiHead(LN1(X))，y=u+MLP(LN2(u))；MLP逐位置C→4C→C，非线性用GELU。多头负责从可见位置聚合信息，MLP逐位置处理特征；残差两边形状一致，LayerNorm按末特征维归一。因果mask不提供完整位置编码，Block不是完整GPT。

**作业建议：`lessons/e07/main.py`**。保持自己编写核心组合逻辑；允许复制自己已验收的SingleHead，或以适当方式引用，无需强制重构原文件。模块形式可按个人习惯调整，以下为实现框架而非额外文件/测试数量要求：

| 功能块 | 目的、输入输出与操作 | API使用与自查 |
|---|---|---|
| 复用两个头 | 两个独立SingleHead均接收[2,4,8]，分别返回[2,4,4]输出及[B,T,T]权重 | `nn.ModuleList(模块序列)`注册并可迭代模块，普通list不会自动注册；不能重复使用同一实例造成权重共享；取元组中的输出参与拼接 |
| 合并和投影 | 沿特征轴拼接两个[B,T,4]得到[B,T,8]，再映射[B,T,8] | `torch.cat(tensors,dim=-1)`返回拼接Tensor；`nn.Linear(8,8)`作用于末轴并注册参数；其他轴不变 |
| Pre-LN及MLP | LN1后多头，与原X相加得u；LN2(u)经8→32→8 MLP再与u相加，输出[2,4,8] | 两个 `nn.LayerNorm(8)` 按最后8维归一并有可学习参数；`nn.Linear`和`nn.GELU`逐位置处理特征；两条残差相加的Tensor形状相同 |
| 验证 | CPU float32随机输入，输出同shape且有限；标量loss反向检查两头和MLP梯度存在有限；改变末token后比较此前输出 | backward累积梯度；`detach().clone()`产生独立对照副本，`torch.no_grad()`关闭比较阶段梯度记录，`torch.allclose`比较浮点输出；不要求末token输出必须改变或全部数值梯度非零 |

主流程伪代码：构造输入→初始化两个独立头及Block→计算 LN1→多头聚合→第一条残差→LN2→MLP→第二条残差→检查shape/有限梯度→另改末token检查前面输出不变。第一步先复用头并完成并行输出、拼接和投影。无需Dropout、位置编码、tokenizer、训练或额外工程；后续再组完整GPT。对应nanoGPT Block的Pre-LN结构，其作者多头实现使用向量化QKV，本课用独立头便于理解。

**E09–E10 本轮组织（2026-11-12）**：把学生E03字符映射/分段与E07因果Block组合成可训练的字符级GPT。建议 `lessons/e09/main.py`；CPU float32模型，输入/标签torch.long；复用原文 `hello world\n` 重复8次（96字符、V=9），按字符位置先分前72训练、后24验证，各自构造T=8的窗口和右移标签，得到[64,8]及[16,8]。相邻窗口高度重叠、语料重复，即使分段验证也不能证明真实泛化。字符词表从训练段构造并确认覆盖验证段。模型 token Embedding(9,16)+position Embedding(8,16)，一个或多个学生已实现Block（可取1层，2头各8维），末端LayerNorm(16)及Linear(16,9)输出[batch,8,9]原始logits；在每个位置预测下一个token。权重绑定、特殊初始化、生成和checkpoint不作为本课门槛。

| 功能块/目的 | 输入输出、操作与关键API/自查 |
|---|---|
| 分段与错位样本 | 先分字符、训练段建词表并编码；长度L的段各取L−T个样本，输入字符i..i+7，标签i+1..i+8；`torch.tensor(...,dtype=torch.long)`得CPU IDs，核对首尾样本、x[:,1:]=y[:,:-1] |
| 表示与Block | IDs [B,8]→token Embedding [B,8,16]；`torch.arange(T,device=idx.device)`为位置long [8]，position Embedding→[8,16]，加法按batch广播→[B,8,16]；复用Block输出同形。Embedding注册可学习表，输入为索引，输出为对应向量；位置与token都接收梯度 |
| 词表输出与loss | 最终LayerNorm与Linear(16,9)得logits [B,8,9]；reshape为[B*8,9]，目标reshape为[B*8] long，`F.cross_entropy`接原始logits和合法目标ID，默认返回平均标量且内部计算log-softmax；不先softmax，不把时间与词表轴混淆 |
| 训练和验证 | 一个CPU全批次短训练：seed42、AdamW lr=.01、weight_decay=0，建议200次更新；train模式前向、zero_grad、backward、step；记录初次及训练后的训练loss，eval/no_grad评估验证loss。检查梯度有限、loss有下降、数据错位与输出shape，验证集不参与step；未下降时调整学习率或步数并诚实记录 |

主流程伪代码：按字符分语料→建立词表和错位窗口→构建token/position embeddings及复用Block→每个位置输出词表logits→展平并计算CE→短训练→eval/no_grad计算训练与验证loss。第一步核对训练/验证各自的首个错位窗口；函数化、层数和打印形式可自选。实验不要求固定数值、泛化质量、生成、下载数据或训练GPU。

### P-E：mini-GPT 必须满足的验收要求

- 自己设计一个**与参考 nanoGPT 源码不逐行照抄**的小型项目。
- 有 Tokenizer、dataset/batch 采样、Causal Self-Attention、Block、GPT、训练、评估、生成、保存/恢复。
- 含可运行的小数据实验；`train loss` 下降，验证损失可报告；固定 seed 可复现关键输出（容忍设备非完全确定性）。
- 单元测试涵盖：tokenizer round trip、mask 的因果性、形状/梯度、采样范围、保存-恢复一致性。
- 能向导师解释：为什么输出 logits 的 Shape 是 `(B,T,V)`；`cross_entropy` 如何组织输入；`QK^T / sqrt(d_k)`；为什么未来信息泄露是错误；参数量的来源。
- 能说清参考项目中入口文件、模型文件、配置、数据预处理之间的关系。
- 若从零训练资源有限，优先用 toy 数据和小模型，而非盲目追大模型效果。

### E 阶段完成后的知识延伸

Encoder-only / Encoder-Decoder；Cross-Attention；RoPE 细节；RMSNorm / SwiGLU；GQA/MQA；LoRA/PEFT；预训练与微调；ViT/DiT。先打通原理再使用框架封装 API。

## 9. 阶段 F：生成建模统一视角（F01–F09）

**目标**：建立密度建模、潜变量、对抗学习、变分推断和概率流的共同框架，防止只会记模型名字。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| F01 | 生成建模：显式似然/隐式生成、采样与推理 | 用toy分布做概率/采样计算并比较任务 |
| F02 | Autoencoder：Encoder/Decoder/重建损失 | 在低维或简化图像数据上重建 |
| F03 | VAE：ELBO、重参数化、KL、潜变量 | 手推并实现基本 VAE |
| F04 | β-VAE、Posterior Collapse、潜变量插值 | 分析重建-正则权衡 |
| F05 | GAN：Generator/Discriminator、minimax | 合成二维数据训练 |
| F06 | GAN 稳定性：mode collapse、判别器失衡 | 诊断失败现象 |
| F07 | Normalizing Flow、变量代换、Jacobian | 用简单可逆映射算密度 |
| F08 | Coupling Layers / RealNVP、可逆性 | 训练二维密度模型 |
| F09 | 对比 VAE/GAN/Flows/AR/扩散与流匹配 | 调用本人已写采样器做小型对照，再整理公式/代码/训练/采样映射 |

**验收 P-F**：VAE + 2D Normalizing Flow 两个小实现；清楚区分 encoder 近似后验、flow 可逆映射、生成时从哪个分布采样；GAN 可作为附加小实验。

## 10. 阶段 G：Diffusion / Score-based 模型（G01–G12）

**目标**：能解释并独立实现前向加噪、噪声预测、反向采样，理解离散与连续时间方法。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| G01 | 马尔可夫链、Gaussian 噪声、时间步、SNR | 画不同 t 的噪声强度 |
| G02 | DDPM 前向过程，β/α/ᾱ 调度 | 直接采样 `x_t` 并验证方差 |
| G03 | 噪声预测 εθ、timestep embedding、MSE | 做 toy ε-prediction |
| G04 | DDPM 训练目标和推导（从 ELBO 到 simplified loss） | 对应每个公式到代码 |
| G05 | DDPM 反向步骤、系数、随机项、边界 | 小规模生成 2D 数据 |
| G06 | 用简化 U-Net / MLP 做图像或二维实验 | 实现条件时间网络 |
| G07 | DDIM 与确定性/非确定性采样 | 对比生成步数和误差 |
| G08 | Score matching 与去噪分数估计 | 解释 score 与 ε 的关系 |
| G09 | VP/VE SDE、概率流 ODE（直观为主） | 映射离散/连续时间记号 |
| G10 | Conditional Diffusion、Classifier-free Guidance | 条件丢弃与采样引导 |
| G11 | Latent Diffusion、VAE、UNet、DiT 概念 | 读图画出各模块数据流 |
| G12 | Scheduler 抽象、采样验证、速度/质量评估 | 对照实现并做消融 |

**关键最小公式（理解后必须能独立写出，而非照搬）**：

- `α_t = 1 - β_t`，`ᾱ_t = ∏_{s=1}^t α_s`。
- `x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε`，其中 `ε ~ N(0,I)`。
- 简化噪声预测目标：`E_{x0,t,ε} ||ε - εθ(x_t,t)||²`。

**验收 P-G**：独立实现 toy DDPM 的前向扩散、训练和采样；检查 t=0 与最大 t 的边界、均值方差、模型输出 Shape、采样无梯度；画出不同时间步的加噪过程；比较不同采样步数。扩展到简化图像任务是进阶选项。

## 11. 阶段 H：Flow Matching、Rectified Flow 与连续时间生成（H01–H11）

**目标**：真正搞清楚“学向量场”和“数值求解 ODE”，理解与 Diffusion 的联系与区别。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| H01 | 常微分方程 ODE、速度场、轨迹、初值问题 | 一维 Euler 积分并检查误差 |
| H02 | Euler、Heun、RK4、步长与稳定性 | 比较解的精度和计算次数 |
| H03 | Continuity Equation / Probability Flow 直觉 | 区分点轨迹与概率密度变化 |
| H04 | 概率路径、线性插值路径、噪声→数据 | 在二维绘制样本轨迹 |
| H05 | Conditional Flow Matching：目标速度场与回归损失 | 实现二维 CFM 训练 |
| H06 | 推理：学习到的向量场 + ODE solver | 从标准 Gaussian 生成二维样本 |
| H07 | Coupling 的选择：独立/OT 配对（入门） | 比较路径弯曲、求解步数 |
| H08 | Rectified Flow：直线路径、速度预测与再流化概念 | 对比普通 CFM 设置 |
| H09 | Diffusion 与 Flow Matching 的概率路径/参数化关联 | 比较 ε/x0/v/velocity 参数化 |
| H10 | Conditional Flow / CFG / Time Conditioning | 添加类别条件与测试 |
| H11 | Flow-based Transformer / DiT 研究阅读入门 | 提取主要模块和 Loss |

**入门公式（以选定的线性条件路径为例，不代表所有 FM 方法）**：

- 取 `x_0 ~ p_0` 为噪声，`x_1 ~ p_data` 为数据；设 `x_t = (1-t)x_0 + t x_1`。
- 对应条件目标速度 `u_t = x_1 - x_0`。
- 训练 `L = E_{t,x0,x1} ||vθ(x_t,t) - (x_1-x_0)||²`。
- 生成通过 ODE `dx_t/dt = vθ(x_t,t)` 从 `t=0` 积分到 `t=1`。

强调：路径分布、耦合方式和时间参数化会影响目标与行为；不得把任意 Flow Matching、Rectified Flow、Diffusion 偷换成“完全一样”。

**验收 P-H**：用小 MLP 在二维 mixture of Gaussians / two moons 等合成数据上训练；自己写 Euler 与 Heun 采样并作分布、轨迹、NFE（函数评估次数）对比；解释为什么 `x_t` 与 target velocity 的定义必须匹配。

## 12. 阶段 I：重要研究算法扩展（I01–I12，按研究兴趣选修）

这些不是“毕业前全部必须做”，而是获得迁移能力的技术池。优先选能支撑用户研究项目的方向。

| 专题 | 核心算法 / 应会实现的最小模块 | 与基础阶段的连接 |
|---|---|---|
| I01 | Contrastive Learning：InfoNCE、SimCLR | 交叉熵、特征归一化、正负样本 |
| I02 | CLIP 风格图文对比学习 | 多模态对齐、双塔网络 |
| I03 | Vision Transformer / DiT | Attention 与 Patchify、AdaLN |
| I04 | LoRA / PEFT / Adapter | 矩阵低秩分解、冻结参数 |
| I05 | Mixture of Experts | Router、top-k、负载均衡 |
| I06 | State Space / Mamba 概念 | 递归、序列模型替代方案 |
| I07 | GNN：Message Passing | scatter / index_add / 邻接关系 |
| I08 | NeRF：位置编码与体渲染基本积分 | MLP、数值积分、相机射线 |
| I09 | 3DGS：Gaussian 参数、投影、alpha compositing | 多视角、渲染、几何约束 |
| I10 | 人体参数化 / 变形场 / Cloth dynamics | SMPL-X、pose conditioning、时空表示 |
| I11 | RL 基础：MDP、Policy Gradient、PPO 概念 | autograd、采样、优化 |
| I12 | 模型参数生成/Hypernetwork/Weight-space | MLP 权重、permutation、latent 建模 |

选修专题的验收：独立实现一个**可验证的小核**，并对一篇开源论文给出函数级阅读笔记。不强迫复刻高算力的整篇论文。

## 13. 阶段 J：真正能读懂陌生项目（纵向贯穿，最后综合验收）

**阅读一个陌生开源项目时统一流程：**

1. 确定任务定义：输入/输出/训练数据/评价指标/论文贡献。
2. 先看 `README`、项目目录、安装文件 `pyproject.toml`/`requirements.txt`，再找运行入口。
3. 从命令行入口追踪配置 → 数据读取 → dataset/dataloader → model 构造 → forward → loss → backward/optimizer → checkpoint → inference。
4. 记录重要类和函数的**调用图、传参含义、返回结构、Shape 表**。
5. 选一个核心模块，把论文公式逐行映射到代码；分辨“方法本身”和“工程/速度优化”。
6. 写一个不依赖原训练集的小单元测试，验证关键模块。
7. 修改一个小参数或功能，预测结果，再运行检查。
8. 针对不理解的 Python/PyTorch 语法开一张补修卡，补齐后返回。

### 源码阅读训练梯度

- J01：教学级 MLP/CNN 仓库（能画训练循环）。
- J02：nanoGPT（能独立复述每个关键模块）。
- J03：官方 PyTorch Transformer 示例 / 简单 ViT（处理真实模块拆分）。
- J04：小型 DDPM / Diffusers 训练示例（理解 scheduler 与 loss）。
- J05：Flow Matching 实现（理解 `sample t → construct x_t → target velocity → optimizer`）。
- J06：学生研究方向中的真实复杂项目（3DGS、人体 avatar、模型参数生成等），只选**一个**核心算子做深读。

**综合毕业任务 P-J**：任意选一篇近年的开源论文，在不让 Agent 代写算法的条件下，独立产出：（a）1 页算法摘要；（b）重要公式与文件/函数对照；（c）端到端数据流和 Shape；（d）一个最小复现模块；（e）测试与消融；（f）README 和限制说明。

## 14. 贯穿课程的工程技能清单

| 能力 | 最迟何时引入 | 具体掌握标准 |
|---|---|---|
| Conda / pip / 版本与 CUDA | 阶段 A | 能定位当前 Python 和依赖版本冲突 |
| 目录/包/导入与入口 | 阶段 A | 能组织 3+ 文件的可导入项目 |
| Git 分支/commit/diff/恢复 | 阶段 A–D | 能保护已有更改并回滚单个错误 |
| Debugging (`pdb`/VS Code) | 阶段 A–C | 能设断点，检查 Tensor 中间值 |
| Shape / dtype / device 诊断 | 阶段 C | 能预测 reshape/broadcast 输出 |
| 单元测试与 gradcheck | 阶段 C–D | 能发现错误实现，不只检查能否运行 |
| 训练日志 / Checkpoint | 阶段 D | 能恢复训练并评估 |
| 配置与命令行 | 阶段 A–E | 参数、代码、实验相互分离 |
| AMP、显存与性能分析 | 阶段 D–E | 能用测量论证优化效果 |
| 论文公式→代码映射 | 阶段 E–J | 能说清函数对应的算法步骤 |
| 文档/实验复现 | 全程 | 别人能从 README 理解运行条件 |

**必要的数学能力随课补齐**：线性代数、概率/统计、微积分/链式法则、优化、信息论、常微分方程（在 Flow Matching 前）。不要求先完成整本数学书。

## 15. 每一类任务的通用模板

### 15.1 导师布置任务时应包含

- 任务名称、任务意义与本节目标；明确练习培养哪项能力、为什么现在练、对应什么实际用途。
- 已有前置知识（哪些已验收、哪些本节介绍）。
- 允许修改的文件（明确路径）；不允许触碰哪些已有文件。
- 输入、输出、Tensor Shape、dtype/device（适用时）。
- 函数接口或伪代码；**不提供完整函数体**。
- 少量核心验收条件，重点是运行成功且正确体现本章内容；边界/跨设备验证仅在与本章核心目标直接相关时要求，否则为可选改进。
- 有理解疑点时直接提出针对性问题；学生可在聊天或注释中回答，不强制预测注释或重复完整手算。
- 当学生卡住时先参考的官方文档/API 名称。

### 15.2 学生提交代码后，导师 Review 模板

1. **确认事实**：读到的文件、命令/输出、无法验证的事项。
2. **结果**：通过 / 基本通过需修改 / 暂不通过。
3. **做对的地方**：准确指出行为与理由（不要无原则表扬）。
4. **最重要的问题**：按正确性 → 逻辑 → Shape → API → 工程风格排序。
5. **提示**：解释原理、位置、检查方法及局部修改策略，不默认给可直接运行的局部代码。
6. **核心问题由学生修正并验证**；非核心问题作为可选建议。需要确认理解时直接提问，接受聊天或注释回答，不机械附加新关卡。
7. **更新 `PROGRESS.md`**，仅记录实际完成的项目/验收/问题。

### 15.3 防止“形式上手写、实际上不会”的设计

出现具体理解疑点时，按需选择有效方式；以下不是每项任务必做的验收清单，不强制注释预测、整张量手算或反复变体：

- 给一个数值与 Shape 完全不同的小变体。
- 让学生先预测输出再执行。
- 把一个操作替换为不等价 API，让学生解释错误。
- 让学生画调用链和输入/输出结构。
- 对比可运行但结果错误的实现与正确实现。
- 从空白文件复写“核心 10–30 行”的思路（不要求每次复写整项目）。
- 让学生解释一个 API 的替代方案与副作用。

## 16. 实验验证习惯与算法质量标准

按当前算法的真实风险选择以下检查，不是每节课强制完成全部八项，也不要求固定测试数量：

1. **Shape**：输入/输出满足约定；不正确时尽早报错。
2. **dtype/device**：CPU/GPU、float32/bfloat16、整数标签是否正确。
3. **数值**：无 NaN/Inf，loss 有意义；对概率分布做总和/范围检查。
4. **梯度**：需要更新的参数有梯度；不应更新的参数不参与计算图。
5. **过拟合**：若适用，能拟合单个 batch（发现训练管线问题）。
6. **种子与复现**：固定必要的随机种子，记录版本；知道并非任何 CUDA 算子都严格确定。
7. **边界条件**：例如 t=0、最后一步、空输入、不足一个 batch、mask 全零等。
8. **基准对照**：手写实现先与简单真值/官方算子对齐，再谈性能。

**禁止“loss 降了所以算法一定正确”的判断**：必须用独立性质或可计算真值验证。

## 17. 进度状态、优先级与跨会话交接

### 17.1 进度状态与证据范围不能混淆

- `NOT_STARTED`：尚未系统讲授。
- `EXPLAINED`：讲授/演示过，但没有验收学生独立实现。
- `IN_PROGRESS`：学生已经开始亲手编写、未完成验收。
- `PASSED`：在已明确的任务/设备范围内，学生代码已验证运行正确且体现核心内容；提示依赖、独立解释程度及待补设备实验分别记录，不因此自动等同于全面独立熟练。
- `NEEDS_REVIEW`：曾完成但核心理解存在缺口，需要复习。
- `OPTIONAL`：研究方向选修，暂不影响主线推进。

状态以 `PROGRESS.md` 为准，勿自行“乐观升级”。

### 17.2 每次课程结束必须交接的内容

- 日期、课程 ID、学习了什么。
- 学生**亲手编写或修改**的文件及简述，不要贴整段代码。
- 验收证据（测试命令/结果、Git commit 可选）。
- 现在**确实掌握**与**仍不清楚**的点。
- 仅在发现具体理解缺口时记录少量待复习点，不把固定数量的问题变成每次开课必答关卡。
- 下一个**具体最小任务**（可在 20–60 分钟内完成的粒度，不是整个里程碑）。
- 遗留错误/环境变更/需要注意的本地文件。

### 17.3 每隔一个阶段做一次 Review

- 回顾从头能否独立写最小模块。
- 整理易错知识清单。
- 根据真实掌握水平调整课次，不追求数量。
- 提取 reusable notes、测试和小项目，不复制 AI 成品。

### 17.4 自动记录与跨设备恢复（2026-10-12 学生确认）

- **验收后自动记录**：立即维护 `PROGRESS.md` 的当前状态、证据、待复习点和下一项最小任务，追加实际 Session Log，保留历史；此项已有持续授权，无需反复确认。仅当课程路线或长期规则改变时更新本计划及 `AGENTS.md`，实际进度统一存于 `PROGRESS.md`。
- **每次开课先读进度**：无论是否新会话、是否同一设备，都重新检查磁盘上的最新进度与相关代码；首先简短说明从哪里继续。指导文档中的建立时快照、旧聊天及 Session 模板不能替代最新实际记录。
- **跨设备操作顺序**：当前设备学习结束 → 将学生作业与更新的文档一起 commit、push → 另一设备开课前 pull → 导师重新读取记录并继续。自动维护文档不代表自动运行 Git 同步命令；学生提出执行同步的请求时再处理。
- **检查范围**：本地 Git 状态只能说明当前工作区情况，不能证明远端没有新提交。发现同步冲突时保留双方实际记录，依据代码与验收证据核对；不能仅按更晚日期或更高课次覆盖进度。
- **设备环境**：作业路径使用仓库相对路径；Windows/Linux、CPU/CUDA 等环境分别记录。切换设备后重新核实当前解释器和所需设备，不能把另一设备的成功运行当成本机验证。

## 18. 建立计划当天的历史快照（2026-10-10，仅存档）

本节保留原始首课安排用于追溯，不代表当前待办；恢复课程只能使用PROGRESS.md最新当前状态与实际Session Log。

### 已经讲解，但未验收

- 项目与模块的初步区别；一个学习项目的基础目录。
- Conda 环境的创建、激活命令含义；Conda / pip 的基本定位；使用 PyTorch CUDA wheel 的概念。
- PyTorch `import torch`、Tensor 创建、`shape`、`dtype`、`device`、均值、二维索引、切片、`.to("cuda")`。
- GPT 学习的最初目标与六阶段简版路线（现在扩为本计划）。

### 尚未获得证据

- `lessons/c01/01_tensor.py` 是否存在/已运行。
- `dllab` 的 `torch.cuda.is_available()` 是否返回 True。
- 学生是否真正理解 `mean(dim=0)` vs `mean(dim=1)`、`x.to('cuda')` 是否原地、二维索引。
- Python 高级语法、Autograd、训练循环是否可独立编写。

### 历史首课任务：C01 / Lesson 01（不是当前待办）

学生在 `~/projects/DeepLearningLab/lessons/c01/01_tensor.py` **自己手写**：

1. 创建值为 `[[1.,2.,3.],[4.,5.,6.]]` 的 2×3 浮点 Tensor。
2. 打印张量及 `.shape`、`.dtype`、`.device`。
3. 索引第一行与第二列并解释结果 Shape。
4. 计算按列的平均值，并解释 `mean(dim=0)` 与 `mean(dim=1)` 的区别。
5. 把 Tensor 移动到 CUDA；清楚区分 `x.to('cuda')` 返回新 Tensor 与是否修改 `x` 变量。
6. 自己给一个不同 Shape 的测试例子，并预测结果。

导师的职责是：先检查是否已有文件，**不要直接替学生创建作业答案**；讲必要语法、给小提示、按验收规则 Review。成功后再考虑按进度进入 C02 或穿插 A 系列的 Python 基础补课。

### 初期教学策略（重要）

“第一个作业 C01”只是自然入口；若学生在写代码时卡在 Python 的 `import`、函数、列表、变量绑定或模块组织，应回到 A01–A07 插入微课，**不是重复整套课程或跳过障碍**。此后按数学需求穿插 B 系列。

2026-10-12 学生已明确确认第 3 节的实践主线安排：继续以 C 及后续内容组织教学，A/B 按前置需求穿插、联系并验收；不将本节建立计划当天的历史快照作为当前待办。

## 19. 课程毕业标准（可检验，不是口号）

通过课程后，学生应该能在有限查阅官方文档的前提下：

- 将小神经网络的数学公式翻译为结构清晰的PyTorch模型与训练流程；以正确性和解释能力为准，不设置固定完成时限。
- 不看现成源码，独立重建 Transformer Block 的重要组件，写 Shape / causal mask / gradient 测试。
- 维护自己的 mini-GPT：配置、训练、评价、生成、ckpt、复现均可自行调试。
- 独立实现 toy DDPM 与 toy Flow Matching，解释 loss 与采样器差别，正确比较公式与 Shape。
- 遇到新模型能够找到核心计算图、阅读训练入口并给出最小可验证改动。
- 能清楚说出“我目前不理解什么、我怎样验证自己的假设”，而不是把模型输出当作事实。

毕业不是课程终点，之后可以以“读论文 → 推导 → 实现最小模块 → 写测试 → 对照开源实现”不断循环。

## 20. 官方资料与参考项目（学习时按需使用，非必须立刻阅读）

- Python Tutorial：https://docs.python.org/3/tutorial/
- Python 标准库：https://docs.python.org/3/library/
- Packaging User Guide：https://packaging.python.org/
- NumPy：https://numpy.org/doc/stable/
- PyTorch Tutorials：https://docs.pytorch.org/tutorials/
- PyTorch API：https://docs.pytorch.org/docs/stable/index.html
- micrograd：https://github.com/karpathy/micrograd
- nanoGPT：https://github.com/karpathy/nanoGPT
- Zero to Hero课程与讲义：https://karpathy.ai/zero-to-hero.html 、 https://github.com/karpathy/nn-zero-to-hero
- minGPT教育参考：https://github.com/karpathy/minGPT
- minbpe练习（首个字符GPT后按需）：https://github.com/karpathy/minbpe/blob/master/exercise.md
- 作者资料的核实范围、使用顺序及课程映射：`docs/RESOURCE_MAP.md`。
- Attention Is All You Need：https://arxiv.org/abs/1706.03762
- DDPM：https://arxiv.org/abs/2006.11239
- DDIM：https://arxiv.org/abs/2010.02502
- Flow Matching for Generative Modeling：https://arxiv.org/abs/2210.02747
- Rectified Flow：https://arxiv.org/abs/2209.03003
- Codex AGENTS.md 参考：https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide
- Agent Skills 参考：https://developers.openai.com/api/docs/guides/tools-skills

原则：优先官方文档、原始论文、官方或作者开源仓库；博客/视频用于补充直觉，遇到版本变化时核实当前 API，不把旧文章当作唯一事实。

## 21. 使用方式：如何让新 Agent 接手

将本包文件放到 `~/projects/DeepLearningLab/` 的相应相对路径后，从此目录启动 Codex。你可以对新 Agent 说：

> 请先阅读项目根目录的 `AGENTS.md`、`PROGRESS.md` 当前状态及最近实际Session，再读 `LEARNING_PLAN.md` 对应主题并按需要使用 `deep-learning-mentor` skill。你是我的编程导师，不是代写者。先只读查看本课已有代码，说明当前状态与下一项最小任务；用原理、实现框架、提示和API指导我手写，验收后按真实证据更新记录。不要凭旧聊天或规划快照从头开课。

如果切换回网页 ChatGPT，只需上传 `LEARNING_PLAN.md`、`PROGRESS.md` 和必要代码，就可以较可靠地恢复到当前课程状态。**任何 Agent 都可能丢失会话上下文，磁盘上的进度与代码才是持续交接的依据。**

---

*维护原则：课程大纲低频更新，实际进度高频更新。若新增前沿算法，在阶段 I 扩展即可；不为追逐新论文而破坏 Python/PyTorch/Transformer/Diffusion/Flow Matching 核心链路。*
