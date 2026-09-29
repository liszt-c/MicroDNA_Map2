"""
scripts/train.py - 模型训练入口

支持:
  1. 平衡型染色体 5 折交叉验证全量自动化执行 (--cv)
  2. 单独训练/评估指定染色体 Fold (--fold 1..5)
  3. 保留经典随机拆分模式 (--split-mode random)
  4. 每折严格在当前 Train 染色体上执行多阶段困难负例挖掘 (HNM)
  5. 产出综合指标表 (ACC, AUC, Recall, Precision, F1, MCC) 与权重集合
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHROM_FOLDS,
    CV_MODEL_DIR,
    DEFAULT_BASE_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLOODING_B,
    DEFAULT_GAMMA,
    DEFAULT_HNM_EPOCHS,
    DEFAULT_HNM_KEEP_EASY,
    DEFAULT_HNM_ROUNDS,
    DEFAULT_HNM_THRESHOLD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEP_SIZE,
    DEFAULT_WEIGHT_DECAY,
    LAYER_SIZE,
    MODEL_DIR,
    NUM_CV_FOLDS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.dataloader import (
    ECC_LABEL,
    build_chromosome_cv_loaders,
    build_train_val_loaders,
    subset_labels,
)
from src.hnm import perform_hnm
from src.model import ResNetSelfAttention
from src.utils import setup_logger

logger = setup_logger('train')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, flooding_b: float = 0.0) -> Dict:
    """在 DataLoader 上执行严谨评估，返回各项量化指标"""
    model.eval()
    total_loss, n_batches, n_samples = 0.0, 0, 0
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if flooding_b > 0:
                loss = (loss - flooding_b).abs() + flooding_b

            total_loss += float(loss.item())
            n_batches += 1
            n_samples += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, ECC_LABEL]
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])

    unique_classes = len(np.unique(y_true))
    acc = float(accuracy_score(y_true, y_pred)) if n_samples else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if n_samples else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if n_samples else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if n_samples else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if unique_classes > 1 else 0.0
    auc = float(roc_auc_score(y_true, y_prob)) if unique_classes > 1 else 0.0

    if unique_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        'loss': total_loss / max(n_batches, 1),
        'n': n_samples,
        'acc': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mcc': mcc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def train_single_fold(
    fold_idx: int,
    args,
    device,
    use_amp: bool,
    scaler,
    out_dir: Path,
    ecc_fa: Path,
    other_fa: Path,
    is_cv_mode: bool = False,
) -> Dict:
    """训练单个 Fold 的完整周期 (Base + HNM + 独立测试集评估)"""
    fold_num = fold_idx + 1
    fold_prefix = f"fold{fold_num}_" if is_cv_mode else ""
    writer = SummaryWriter(log_dir=str(out_dir / f"logs_fold{fold_num}"))

    # 1. 建立基于染色体隔离的数据加载器
    train_loader, val_loader, test_loader, full_ds, train_sub, val_sub, test_sub = build_chromosome_cv_loaders(
        test_fold=fold_idx,
        batch_size=args.batch_size,
        seed=args.seed + fold_idx * 100,
        num_workers=args.num_workers,
        balanced=args.balanced,
        ecc_path=ecc_fa,
        other_path=other_fa,
    )

    model = ResNetSelfAttention(layer_size=args.layer_size).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_val_score = -1.0
    best_val_metrics = {}
    best_epoch = 0
    global_epoch_counter = 0

    total_stages = args.hnm_rounds + 1
    best_ckpt_path = out_dir / f"{fold_prefix}best_model.pth"

    # 2. 阶段训练循环 (Base -> 多轮 HNM)
    for stage in range(total_stages):
        is_hnm_stage = (stage > 0)
        epochs_for_stage = args.hnm_epochs if is_hnm_stage else args.base_epochs
        stage_name = f"HNM Round {stage}" if is_hnm_stage else "Base Stage"

        logger.info(f"\n[Fold {fold_num}] --- Starting {stage_name} ({stage + 1}/{total_stages}) ---")

        if is_hnm_stage:
            if best_ckpt_path.exists():
                logger.info(f"[Fold {fold_num}] Loading best checkpoint for HNM resampling: {best_ckpt_path.name}")
                ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['state_dict'])

            # 约束 HNM 仅在当前 Train 染色体上挖掘
            train_loader, train_sub = perform_hnm(
                model=model,
                full_dataset=full_ds,
                current_train_subset=train_sub,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                threshold=args.hnm_threshold,
                keep_easy_ratio=args.hnm_keep_easy,
                balanced=args.balanced,
                seed=args.seed + fold_idx * 100 + stage,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = (
            lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            if args.step_size > 0 else None
        )

        for epoch in range(1, epochs_for_stage + 1):
            global_epoch_counter += 1
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Fold {fold_num} [{stage_name}] Ep {epoch}/{epochs_for_stage}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if args.flooding_b > 0:
                        loss = (loss - args.flooding_b).abs() + args.flooding_b

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                total += labels.size(0)
                batch_correct = int((outputs.float().argmax(1) == labels).sum().item())
                correct += batch_correct
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_correct / labels.size(0):.3f}"})

            if scheduler is not None:
                scheduler.step()

            # 验证集评估
            vm = evaluate(model, val_loader, criterion, device, flooding_b=0.0)
            cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"[Fold {fold_num} Ep {global_epoch_counter}] lr={cur_lr:.2e} | "
                f"Train Loss={running_loss / max(len(train_loader), 1):.4f} Acc={correct / max(total, 1):.4f} | "
                f"Val Loss={vm['loss']:.4f} Acc={vm['acc']:.4f} AUC={vm['auc']:.4f} F1={vm['f1']:.4f}"
            )

            # 以验证集 AUC (若未分化则退化为 ACC) 选拔最优模型
            val_score = vm['auc'] if vm['auc'] > 0 else vm['acc']
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = global_epoch_counter
                best_val_metrics = vm.copy()

                torch.save({
                    'epoch': global_epoch_counter,
                    'fold': fold_num,
                    'test_chromosomes': CHROM_FOLDS[fold_idx],
                    'state_dict': model.state_dict(),
                    'val_acc': vm['acc'],
                    'val_auc': vm['auc'],
                    'val_f1': vm['f1'],
                    'layer_size': args.layer_size,
                    'label_convention': 'eccDNA=1, otherDNA=0',
                }, best_ckpt_path)
                logger.info(f"  -> [Fold {fold_num}] Saved NEW BEST Checkpoint (Val AUC={vm['auc']:.4f}, Acc={vm['acc']:.4f})")

    writer.close()

    # 3. 最终评测: 加载该 Fold 验证集最优权重，评估完全独立的 Held-out Test Fold
    logger.info(f"\n[Fold {fold_num}] Evaluating Held-out Test Set using Best Weights (from Epoch {best_epoch})...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device, flooding_b=0.0)

    logger.info(
        f"[Fold {fold_num} Final Test] Held-out Chroms: {CHROM_FOLDS[fold_idx]} | "
        f"ACC={test_metrics['acc']:.4f}, AUC={test_metrics['auc']:.4f}, "
        f"Recall={test_metrics['recall']:.4f}, Precision={test_metrics['precision']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}"
    )

    return {
        'fold': fold_num,
        'test_chroms': ",".join(CHROM_FOLDS[fold_idx]),
        'best_epoch': best_epoch,
        'val_auc': best_val_metrics.get('auc', 0.0),
        'val_acc': best_val_metrics.get('acc', 0.0),
        'test_acc': test_metrics['acc'],
        'test_auc': test_metrics['auc'],
        'test_recall': test_metrics['recall'],
        'test_precision': test_metrics['precision'],
        'test_f1': test_metrics['f1'],
        'test_mcc': test_metrics['mcc'],
        'test_tp': test_metrics['tp'],
        'test_fp': test_metrics['fp'],
        'test_tn': test_metrics['tn'],
        'test_fn': test_metrics['fn'],
        'model_path': str(best_ckpt_path),
    }


def main():
    p = argparse.ArgumentParser(
        description='Train MicroDNA classifier with Balanced Chromosome-grouped 5-Fold Cross-Validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--cv', action='store_true', help="启用 5 折染色体交叉验证全流程")
    p.add_argument('--fold', type=int, default=None, help="仅训练指定的单个染色体 Fold (1 到 5)")
    p.add_argument('--split-mode', choices=['chromosome', 'random'], default='chromosome',
                   help="数据集拆分模式: 'chromosome' (平衡染色体隔离，默认推荐) 或 'random' (经典随机打乱)")
    
    p.add_argument('--base-epochs', type=int, default=DEFAULT_BASE_EPOCHS)
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE)
    p.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    p.add_argument('--flooding-b', type=float, default=DEFAULT_FLOODING_B)
    p.add_argument('--balanced', action='store_true', help="训练集启用类别加权平衡采样")
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--layer-size', type=int, default=LAYER_SIZE)

    # 困难负例挖掘参数
    p.add_argument('--hnm-rounds', type=int, default=DEFAULT_HNM_ROUNDS)
    p.add_argument('--hnm-epochs', type=int, default=DEFAULT_HNM_EPOCHS)
    p.add_argument('--hnm-threshold', type=float, default=DEFAULT_HNM_THRESHOLD)
    p.add_argument('--hnm-keep-easy', type=float, default=DEFAULT_HNM_KEEP_EASY)

    p.add_argument('--ecc-excel-fasta', dest='ecc_fa', default=None)
    p.add_argument('--other-fasta', dest='other_fa', default=None)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    if torch.cuda.is_available():
        gpu_cap = torch.cuda.get_device_capability()
        use_amp = (gpu_cap[0] >= 7)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ecc_fa = Path(args.ecc_fa) if args.ecc_fa else PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = Path(args.other_fa) if args.other_fa else PROCESSED_DATA_DIR / "otherDNA.fa"

    # =========================================================================
    # 模式 A: 平衡型染色体 5 折交叉验证模式 (--cv)
    # =========================================================================
    if args.cv or (args.split_mode == 'chromosome' and args.fold is None and not args.output_dir):
        out_dir = Path(args.output_dir) if args.output_dir else CV_MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== Starting Balanced Chromosome-grouped 5-Fold Cross-Validation ===")
        logger.info(f"Output directory: {out_dir}")

        all_fold_records: List[Dict] = []
        for f_idx in range(NUM_CV_FOLDS):
            logger.info(f"\n{'#'*75}\n### Cross-Validation: Processing Fold {f_idx + 1}/{NUM_CV_FOLDS} ###\n{'#'*75}")
            res = train_single_fold(
                fold_idx=f_idx, args=args, device=device, use_amp=use_amp, scaler=scaler,
                out_dir=out_dir, ecc_fa=ecc_fa, other_fa=other_fa, is_cv_mode=True
            )
            all_fold_records.append(res)

        # 汇总 5 折指标计算均值与标准差
        df_cv = pd.DataFrame(all_fold_records)
        metric_cols = ['test_acc', 'test_auc', 'test_recall', 'test_precision', 'test_f1', 'test_mcc']
        mean_series = df_cv[metric_cols].mean()
        std_series = df_cv[metric_cols].std()

        summary_rows = []
        for r in all_fold_records:
            summary_rows.append({
                'Fold': f"Fold {r['fold']}",
                'Test_Chromosomes': r['test_chroms'],
                'Accuracy': f"{r['test_acc']:.4f}",
                'AUC_ROC': f"{r['test_auc']:.4f}",
                'Recall': f"{r['test_recall']:.4f}",
                'Precision': f"{r['test_precision']:.4f}",
                'F1_Score': f"{r['test_f1']:.4f}",
                'MCC': f"{r['test_mcc']:.4f}",
            })

        summary_rows.append({
            'Fold': 'Mean',
            'Test_Chromosomes': 'ALL',
            'Accuracy': f"{mean_series['test_acc']:.4f}",
            'AUC_ROC': f"{mean_series['test_auc']:.4f}",
            'Recall': f"{mean_series['test_recall']:.4f}",
            'Precision': f"{mean_series['test_precision']:.4f}",
            'F1_Score': f"{mean_series['test_f1']:.4f}",
            'MCC': f"{mean_series['test_mcc']:.4f}",
        })
        summary_rows.append({
            'Fold': 'Std',
            'Test_Chromosomes': '-',
            'Accuracy': f"{std_series['test_acc']:.4f}",
            'AUC_ROC': f"{std_series['test_auc']:.4f}",
            'Recall': f"{std_series['test_recall']:.4f}",
            'Precision': f"{std_series['test_precision']:.4f}",
            'F1_Score': f"{std_series['test_f1']:.4f}",
            'MCC': f"{std_series['test_mcc']:.4f}",
        })

        summary_df = pd.DataFrame(summary_rows)
        csv_path = out_dir / "cv_summary_metrics.csv"
        tsv_path = out_dir / "cv_summary_metrics.tsv"
        summary_df.to_csv(csv_path, index=False)
        summary_df.to_csv(tsv_path, sep="\t", index=False)

        print("\n" + "=" * 90)
        print("          BALANCED CHROMOSOME 5-FOLD CROSS-VALIDATION FINAL REPORT")
        print("=" * 90)
        print(summary_df.to_string(index=False))
        print("=" * 90)
        logger.info(f"Cross-Validation Summary Report exported to {csv_path}")

        return {
            'best_auc': float(mean_series['test_auc']),
            'best_acc': float(mean_series['test_acc']),
            'std_auc': float(std_series['test_auc']),
            'std_acc': float(std_series['test_acc']),
        }

    # =========================================================================
    # 模式 B: 运行单个指定的染色体 Fold (--fold 1..5)
    # =========================================================================
    elif args.fold is not None:
        fold_idx = int(args.fold) - 1
        if not (0 <= fold_idx < NUM_CV_FOLDS):
            logger.error(f"Invalid fold index {args.fold}. Must be between 1 and {NUM_CV_FOLDS}.")
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR / f"fold_{args.fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = train_single_fold(
            fold_idx=fold_idx, args=args, device=device, use_amp=use_amp, scaler=scaler,
            out_dir=out_dir, ecc_fa=ecc_fa, other_fa=other_fa, is_cv_mode=False
        )
        return {'best_auc': res['test_auc'], 'best_acc': res['test_acc']}

    # =========================================================================
    # 模式 C: 经典随机划分模式 (--split-mode random)
    # =========================================================================
    else:
        out_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader, full_ds, train_sub, val_sub = build_train_val_loaders(
            batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
            balanced=args.balanced, ecc_path=ecc_fa, other_path=other_fa
        )
        model = ResNetSelfAttention(layer_size=args.layer_size).to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        best_auc, best_acc = 0.0, 0.0

        for epoch in range(1, args.base_epochs + 1):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            vm = evaluate(model, val_loader, criterion, device)
            if vm['auc'] > best_auc:
                best_auc, best_acc = vm['auc'], vm['acc']
                torch.save({'state_dict': model.state_dict(), 'layer_size': args.layer_size}, out_dir / "best_model.pth")

        return {'best_auc': best_auc, 'best_acc': best_acc}


if __name__ == '__main__':
    main()