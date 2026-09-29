# MicroDNA Map v2.0

MicroDNA Map v2.0 是一个用于染色体外环状 DNA（eccDNA / microDNA）识别、注释与基准评测的计算平台。该平台结合了一维 ResNet-SelfAttention 深度学习网络、双层滑动窗口聚合机制，以及本项目自研的微尺度局部覆盖度分析算法（`micro_coverage`），专门针对 microDNA 物理尺度特征进行富集区捕获。

---

## 项目结构

```text
MicroDNA_Map/
├── config.py                  # 全局配置中心与超参数定义
├── requirements.txt           # Python 依赖清单
├── data/                      # 原始数据与处理后序列目录
│   ├── raw/                   # 原始坐标表格与 FASTQ 测序数据
│   └── processed/             # 处理后的标准 FASTA (eccDNA.fa, otherDNA.fa)
├── refs/                      # 参考基因组与相关索引
├── models/                    # 模型权重与训练日志
│   └── cv/                    # 5 折交叉验证权重与汇总指标评估
├── results/                   # 推理预测、中间过程与评估指标
├── src/                       # 核心算法源码
│   ├── model.py               # ResNetSelfAttention 网络模型结构
│   ├── dataloader.py          # 惰性二进制偏移索引 Dataset 与染色体平衡 5 折交叉验证分流器
│   ├── dataprocess.py         # 序列清洗、One-hot 向量化与 FASTA 解析
│   ├── hnm.py                 # 多轮在线困难负例挖掘引擎
│   ├── utils.py               # 通用工具函数与外部子进程封装
│   └── pipeline/              # 候选区域初筛流水线
│       ├── micro_coverage_pipeline.py # 本项目自研微尺度覆盖度扫描流程
│       └── cnvkit_pipeline.py         # 传统宏观 CNVkit 分析流程
├── scripts/                   # 业务与运行脚本
│   ├── process_data.py        # 从坐标表提取并构建标准数据集
│   ├── sample_negatives.py    # 基因组背景负例随机采样补充
│   ├── train.py               # 平衡型染色体 5 折交叉验证训练与多阶段 HNM / AMP
│   ├── verify.py              # 单折独立测试染色体评估与 5 折集成量化分析
│   ├── compare_hnm.py         # HNM 概率分布变化与长尾抑制分析
│   ├── predict.py             # 高通量批量推理
│   ├── batch_process.py       # FASTQ 双端测序端到端全流程分析
│   └── ablation_layer_size.py # 模型基础通道宽度消融实验
└── benchmark/                 # Spike-in 仿真、工具检测与对比基准
    ├── config_real.yaml       # 真实序列 Spike-in 评测配置
    ├── config_random.yaml     # 随机序列 Spike-in 评测配置
    ├── run_benchmark.py       # 10 阶段基准测试自动化流水线
    ├── microdna_junction_rescuer.py # 环状 Head-to-Tail 嵌合接头读段挽救模块
    ├── perturbation_test.py   # 特征依赖性扰动测试与饱和突变分析
    ├── models_comparison/     # 经典对比模型架构 (ResNet50, Transformer 等)
    ├── simulate/              # Circle-seq 与 WGS 模拟读段生成器
    ├── detect/                # 各检测软件封装脚本 (MicroDNA Map, Circle-Map)
    ├── evaluate/              # 重叠度判定与性能指标计算
    └── visualize/             # 基准测试指标整合可视化工具

```

---

## 环境安装

创建隔离的运行环境并安装相关依赖库：

```bash
conda create -n microdna python=3.9 -y
conda activate microdna
pip install -r requirements.txt

# 安装与宿主机 CUDA 版本匹配的 PyTorch (以 CUDA 12.1 为例)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install seaborn cnvkit pysam openpyxl

```

运行端到端流程与基准测试时，需确保以下外部工具已安装并加入系统环境变量 `PATH`：

* `bowtie2` 与 `bowtie2-build`
* `bwa`
* `samtools`
* `cnvkit.py`（可选，仅在使用原生 CNVkit 流程时需要）
* `Circle-Map`（可选，仅在运行基准测试对比时需要）

---

## 快速使用

### 1. 序列推断预测 (`scripts/predict.py`)

使用 `predict.py` 直接对给定的 FASTA 序列文件进行打分或滑动窗口扫描检测。支持单模型权重以及 5 折交叉验证权重集成软投票（Soft Ensemble）。

常用参数说明：

* `--input`：输入的 FASTA 文件路径或包含序列文件的目录。
* `--mode`：推断模式。`short` 用于定长 400 bp 序列单次批量分类；`long` 用于长序列滑动窗口扫描与平滑聚合。默认：`long`。
* `--model`：模型权重文件路径（`.pth`）或包含 5 折权重的目录（如 `models/cv`）。默认：`models/cv` 或 `models/best_model.pth`。
* `--limit`：判定为 eccDNA 的概率阈值。默认：`0.75`。
* `--batch-size`：推理批处理大小。默认：`512`。
* `--min-region-len`：长序列模式下保留的最小预测区域长度（bp）。默认：`150`。
* `--no-merge`：禁用对相邻重叠候选区域的自动合并。
* `--output-dir`：结果输出目录。

```bash
# 短序列模式：输出包含序列标识符与分类概率的 TSV 表格
python scripts/predict.py --input data/test_short.fa --mode short --batch-size 512

# 长序列模式 (单模型)：滑窗扫描并生成候选区域 BED 与对应 FASTA
python scripts/predict.py --input data/test_long.fa --mode long --model models/best_model.pth --limit 0.75 --output-dir results/predictions

# 长序列模式 (5 折集成软投票)：平滑跨染色体方差，获得最高泛化稳定性
python scripts/predict.py --input data/test_long.fa --mode long --model models/cv --limit 0.75 --output-dir results/predictions

```

### 2. 基因组直接扫描 (`benchmark/detect/run_microdna_map_direct.py`) [不推荐使用]

> **注意：** 在缺乏测序读段覆盖度初筛的情况下，对全染色体直接进行滑窗扫描会带来极大的计算资源消耗和较高的假阳性风险。该脚本主要用于理论消融实验与方法学对比；对于常规测序样本的鉴定分析，推荐使用端到端流程 `scripts/batch_process.py`。

无需进行读段比对与覆盖度分析，直接将参考基因组分段切块后调用模型滑动窗口扫描。

常用参数说明：

* `--reference`：参考基因组 FASTA 文件路径。
* `--output_dir`：结果输出目录。
* `--model_path`：模型文件路径。默认：`models/6.pth`。
* `--limit`：识别判别阈值。默认：`0.99`。
* `--region`：指定扫描区域或染色体（例如 `chr21` 或 `chr21:10000000-20000000`），支持重复多次传递。
* `--region_file`：通过 BED 文件指定待扫描区域列表。
* `--segment_length`：内存切块流式处理的片段长度（bp）。默认：`1000000`。

```bash
python benchmark/detect/run_microdna_map_direct.py \
    --reference refs/hg19.fa \
    --output_dir results/direct_scan \
    --region chr21 --region chr22 \
    --limit 0.9

```

### 3. 端到端全流程分析 (`scripts/batch_process.py`)

使用 `batch_process.py` 从原始双端 FASTQ 测序数据开始，依次执行比对、变异初筛与深度学习精准判别。

本项目自研的微尺度局部覆盖度分析流程（`micro_coverage`）为默认初筛后端，专为 microDNA 尺度（150–1000 bp）设计，极大降低了传统宏观工具的高假阳性率；同时保留 `cnvkit` 作为可选对比后端。

常用参数说明：

* `--input-dir`：包含配对 FASTQ 文件的目录。
* `--pipeline`：候选区域初筛流程后端。可选 `micro_coverage`（本项目自研微尺度预扫描，默认推荐）或 `cnvkit`（传统宏观 CNV 调用）。
* `--model`：深度学习模型权重路径（支持传入 `.pth` 单权重或 `models/cv` 目录启用集成投票）。
* `--threads`：运行线程数。
* `--limit`：深度学习判别阈值。默认：`0.75`。
* `--window-size`：`micro_coverage` 模式下的窗口大小（bp）。默认：`200`。
* `--fold-change`：`micro_coverage` 模式下的局部富集倍数阈值。默认：`1.3`。
* `--cluster-max-len`：富集窗口聚合成簇的最大容忍跨度（bp）。默认：`5000`。
* `--min-cnv-size` / `--max-cnv-size`：变异片段长度过滤限制。
* `--cleanup`：运行完成后自动删除中间 SAM 等临时文件。
* `--keep-bam`：在开启 `--cleanup` 时保留排序后的 BAM 及其索引。
* `--output-dir`：最终 BED 与 FASTA 结果输出目录。

```bash
# 默认流程：采用本项目自研的 micro_coverage 流程
python scripts/batch_process.py --input-dir data/raw --model models/cv --threads 16 --cleanup --keep-bam

# 可选流程：切换为传统 CNVkit 流程
python scripts/batch_process.py --input-dir data/raw --pipeline cnvkit --threads 16

```

---

## 开发者指南

### 1. 数据预处理

使用 `process_data.py` 和 `sample_negatives.py` 构建训练所需的高质量标准数据集。

`scripts/process_data.py`（表格坐标提取）：

* `--input`：输入的 Excel 坐标表（`.xlsx`）。
* `--label`：指定标签种类，可选 `ecc`（对应 `eccDNA`）或 `other`（对应 `otherDNA`）。
* `--mode`：长度调整策略。可选 `expand`（向外扩增至目标长度）、`middle`（取跨度超限片段正中）、`raw`（保持原样）。默认：`expand`。
* `--target-len`：输出序列目标长度（bp）。默认：`400`。
* `--min-span`：针对 `middle` 模式的跨度下限（bp）。默认：`600`。
* `--ffill-chrom`：针对合并单元格导出时染色体缺失的问题进行向下填充。

`scripts/sample_negatives.py`（背景负例采样）：

* `--ratio`：按正样本数量比例自动计算并补齐所需负例数。
* `--count`：未指定 `--ratio` 时的固定采样负例数量。默认：`100000`。
* `--ref`：参考基因组 FASTA 路径。
* `--target-len`：生成的序列长度（bp）。默认：`400`。

```bash
# 提取已知正负样本
python scripts/process_data.py --input data/raw/eccDNA_annotations.xlsx --label ecc --mode expand
python scripts/process_data.py --input data/raw/otherDNA_annotations.xlsx --label other --mode middle --min-span 600

# 扩充基因组随机背景负例
python scripts/sample_negatives.py --ratio 1.2 --ref refs/hg19.fa

```

### 2. 模型训练 (`scripts/train.py`)

使用 `train.py` 启动训练，集成平衡型染色体 5 折交叉验证、多阶段在线困难负例挖掘（HNM）与自适应自动混合精度（AMP）。

常用参数说明：

* `--cv`：启动全自动平衡型染色体 5 折交叉验证全流程。
* `--fold`：仅训练指定的单个染色体 Fold（1 到 5）。
* `--split-mode`：数据集划分策略。可选 `chromosome`（平衡染色体隔离，默认推荐）或 `random`（经典随机打乱）。
* `--base-epochs`：基础训练阶段的轮数。默认：`30`。
* `--hnm-rounds`：困难负例挖掘轮次（0 表示纯基础训练）。默认：`1`。
* `--hnm-epochs`：每轮 HNM 附加训练轮数。默认：`20`。
* `--hnm-threshold`：判断为困难负例的概率阈值下限。默认：`0.1`。
* `--hnm-keep-easy`：保留简单负例的比例设置（防止灾难性遗忘）。默认：`0.1`。
* `--batch-size`：训练批次大小。默认：`256`。
* `--lr`：初始学习率。默认：`0.001`。
* `--layer-size`：ResNet 基础通道数。默认：`8`。
* `--balanced`：启用加权类别平衡采样。
* `--output-dir`：输出权重与指标报告保存目录。

```bash
# 1. 启动全自动平衡型染色体 5 折交叉验证 (推荐)
python scripts/train.py \
    --cv \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/cv

# 2. 单独训练指定染色体 Fold (如 Fold 1)
python scripts/train.py \
    --fold 1 \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/fold_1

# 3. 经典随机拆分模式训练
python scripts/train.py \
    --split-mode random \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/run_random

```

### 3. 模型评估 (`scripts/verify.py`)

使用 `verify.py` 对模型进行量化评估，支持单折独立测试染色体评估与 5 折集成模型全量测试，输出完整指标报告与混淆矩阵图表。

常用参数说明：

* `--model`：模型权重文件路径（`.pth`）或包含 5 折权重的目录（如 `models/cv`）。
* `--fold`：仅在指定 Fold 的独立 Held-out 测试染色体上进行评估（1 到 5）。
* `--threshold`：分类判别阈值。默认：`0.5`。
* `--output-dir`：评估报告与图像保存目录。
* `--no-plot`：仅输出文本指标报告，不生成相关图表。

```bash
# 评估单折模型在其专属的独立测试染色体上的表现 (如 Fold 1)
python scripts/verify.py \
    --model models/cv/fold1_best_model.pth \
    --fold 1 \
    --threshold 0.5 \
    --output-dir results/metrics/fold1

# 使用 5 折集成软投票评估全量数据集
python scripts/verify.py \
    --model models/cv \
    --threshold 0.5 \
    --output-dir results/metrics/ensemble

```

### 4. 困难负例挖掘分析 (`scripts/compare_hnm.py`)

对比基线模型与经过 HNM 训练模型的预测概率密度分布（KDE），展示长尾假阳性的抑制与置信度提升。

常用参数说明：

* `--task`：运行模式。可选 `infer`（生成预测数据 CSV）、`plot`（直接读取 CSV 绘图）、`all`（执行完整任务）。默认：`all`。
* `--model-base`：未进行 HNM 训练的基准权重路径。
* `--model-hnm`：经过 HNM 优化后的模型权重路径。
* `--csv-base` / `--csv-hnm`：在 `plot` 模式下直接读取的 CSV 数据路径。
* `--output-dir`：输出图表与 CSV 文件目录。

```bash
# 完整执行推断与绘图
python scripts/compare_hnm.py \
    --task all \
    --model-base models/base.pth \
    --model-hnm models/best_model.pth

```

---

## 基准测试

基准测试目录 `benchmark/` 包含完整的仿真实验、对比架构与特征依赖性验证体系：

```text
benchmark/
├── config_real.yaml       # 真实数据 Spike-in 测试配置
├── config_random.yaml     # 随机数据 Spike-in 测试配置
├── run_benchmark.py       # 10 阶段基准测试自动化流水线
├── microdna_junction_rescuer.py # 环状 Head-to-Tail 嵌合接头读段挽救模块
├── perturbation_test.py   # 模型特征依赖性序列扰动测试
├── models_comparison/     # 经典对比模型架构 (ResNet50, Transformer 等)
├── simulate/              # Circle-seq 与 WGS 模拟读段生成器
├── detect/                # 各检测软件调用封装脚本
├── evaluate/              # 重叠度计算与精确度评估工具
└── visualize/             # 基准测试指标整合可视化工具

```

### 1. 通道容量消融实验 (`scripts/ablation_layer_size.py`)

自动化测试不同的 ResNet 通道配置，探索最佳模型容量。

常用参数说明：

* `--sizes`：待测试的基础通道大小列表。默认：`2 4 8 16 32 64 128 256 512`。
* `--epochs`：每个配置的训练轮数。默认：`40`。
* `--batch-size`：批次大小。默认：`256`。
* `--balanced`：启用类别平衡采样。
* `--dry-run`：仅预览待运行矩阵而不实际开始训练。

```bash
python scripts/ablation_layer_size.py --sizes 8 16 32 64 --epochs 40 --balanced

```

### 2. 对比模型验证 (`benchmark/models_comparison/train_eval.py`)

在严格统一的数据流与随机划分下训练和评估对比网络。

常用参数说明：

* `--model`：待评测架构。可选 `resnet50`（参数量约 6M）、`resnet_no_att`（去除注意力的消融模型）、`transformer`（8 层 depth，256 维 embedding）。
* `--epochs`：训练轮数。默认：`30`。
* `--batch-size`：批次大小。默认：`256`。

```bash
python benchmark/models_comparison/train_eval.py --model resnet50 --epochs 30
python benchmark/models_comparison/train_eval.py --model resnet_no_att --epochs 30
python benchmark/models_comparison/train_eval.py --model transformer --epochs 30

```

### 3. 序列特征扰动测试 (`benchmark/perturbation_test.py`)

通过保持 GC 含量恒定的序列打乱与单碱基饱和突变，验证模型预测是否真正依赖局部序列基序（Motif）。

常用参数说明：

* `--task`：运行模式。可选 `infer`、`plot`、`all`。默认：`all`。
* `--model`：评估模型权重路径。
* `--input`：用于测试的真实正例微环序列集合（FASTA）。
* `--shuffling-n`：打乱测试的序列数量。默认：`5000`。
* `--mutagenesis-n`：饱和突变测试的序列数量。默认：`500`。

```bash
python benchmark/perturbation_test.py \
    --task all \
    --model models/best_model.pth \
    --input data/processed/eccDNA.fa \
    --output-dir benchmark/results/perturbation

```

### 4. Spike-in 仿真实验 (`benchmark/run_benchmark.py`)

执行统一的 Spike-in 仿真全流程管线，覆盖数据模拟、多软件交叉检测与分拷贝数指标评估。

全流程执行阶段编号：

* `Phase 0`：环境与参考索引完整性检查（`.fai`, BWA, Bowtie2）。
* `Phase 1`：生成微环真值集（`generate_microdna.py`）。
* `Phase 2`：模拟 Circle-seq 双端富集测序数据（`simulate_circseq.py`）。
* `Phase 3`：模拟含全基因组背景的 WGS 测序数据（`simulate_wgs.py`）。
* `Phase 4`：在 Circle-seq 数据上运行 Circle-Map 检测。
* `Phase 5`：在 WGS 数据上运行 MicroDNA Map 检测。
* `Phase 6`：在 WGS 数据上运行 Circle-Map 检测（阴性对照）。
* `Phase 7`：在 Circle-seq 数据上运行 MicroDNA Map 检测（跨模态评估）。
* `Phase 8`：量化性能评估（计算 Recall, Precision, F1 与边界误差）。
* `Phase 9`：综合绘图（LOD 曲线、性能对比图）与 Markdown 报告生成。

常用参数说明：

* `--config`：指定 YAML 格式的实验配置定义文件。默认：`benchmark/config_real.yaml`。
* `--phase`：设定运行阶段编号，数字逗号分隔（如 `0,1,2,5,7,8`）。未指定则默认执行全阶段 0–9。
* `--quick`：启用限制检测区域的快速测试模式（仅使用 `chr21` 与 `chr22`）。
* `--dry-run`：仅打印各阶段待执行命令而不实际启动。

```bash
# 快速运行指定测试阶段
python benchmark/run_benchmark.py --config benchmark/config_real.yaml --quick --phase 1,2,5,7,8

# 运行完整基准测试全流程
python benchmark/run_benchmark.py --config benchmark/config_real.yaml

```

### 5. 接头读段挽救与富集检验 (`benchmark/microdna_junction_rescuer.py`)

从 BAM 比对中识别环状 Head-to-Tail 嵌合跨断裂点读段（Split-reads），并在靶向候选区域与协变量（测序深度、片段长度）匹配对照组之间执行 Fisher 精确检验与 Mann-Whitney U 检验。

常用参数说明：

* `-b, --bam`：已排序并建立索引的输入 BAM 文件。
* `-t, --target-bed`：靶向候选 microDNA 区域 BED 文件。
* `-r, --reference`：参考基因组 FASTA 文件（需包含 `.fai` 索引）。
* `-o, --out-prefix`：输出文件命名前缀。默认：`rescued_microdna`。
* `--min-mapq`：主比对与补充比对段的最低 MAPQ 质量值。默认：`20`。
* `--min-circle-len` / `--max-circle-len`：允许挽救的环状分子长度范围（bp）。默认：`150` / `3000`。
* `--depth-tolerance`：背景对照区间深度匹配容忍比率。默认：`0.35`。

```bash
python benchmark/microdna_junction_rescuer.py \
    --bam results/detect/microdna_map/cn10/aligned.sorted.bam \
    --target-bed results/detect/microdna_map/cn10/detected_microdna.bed \
    --reference refs/hg19.fa \
    --out-prefix benchmark/results/rescue_cn10

```