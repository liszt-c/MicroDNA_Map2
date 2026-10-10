#!/usr/bin/env python3
"""
scripts/ablation_layer_size.py - LAYER_SIZE 消融实验一键运行

用法:
  python scripts/ablation_layer_size.py --sizes 8 16 32 64
  python scripts/ablation_layer_size.py --dry-run
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_DIR
from src.utils import setup_logger

logger = setup_logger('ablation')

CSV_COLUMNS = [
    'layer_size', 'batch_size', 'epochs', 'best_epoch',
    'best_val_auc', 'best_val_acc', 'final_val_loss',
    'final_val_acc', 'final_val_auc', 'final_val_f1',
    'model_dir', 'elapsed_sec', 'status',
]

DEFAULT_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ABLATION_ROOT = MODEL_DIR / "ablation_results"
CSV_PATH = ABLATION_ROOT / "ablation_layer_size.csv"


def load_train_main():
    train_script = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_module", str(train_script))
    mod = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    sys.argv = ['train.py']
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = original_argv
    return mod.main


def get_completed_sizes(csv_path: Path) -> set:
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'ok':
                try:
                    completed.add(int(row['layer_size']))
                except (ValueError, KeyError):
                    pass
    return completed


def append_csv_row(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_single_ablation(train_main_fn, layer_size: int, batch_size: int,
                        epochs: int, seed: int, balanced: bool,
                        dry_run: bool = False) -> dict:
    model_dir = ABLATION_ROOT / f"layer_{layer_size}"

    result_row = {
        'layer_size': layer_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model_dir': str(model_dir),
    }

    if dry_run:
        logger.info(f"Dry run: layer_size={layer_size}, bs={batch_size}, output={model_dir}")
        result_row.update({c: '' for c in CSV_COLUMNS if c not in result_row})
        result_row['status'] = 'dry_run'
        return result_row

    logger.info(f"Processing LAYER_SIZE={layer_size} | Batch Size={batch_size} | Epochs={epochs}")
    start_time = time.time()

    try:
        original_argv = sys.argv
        train_argv = [
            'train.py',
            '--layer-size', str(layer_size),
            '--batch-size', str(batch_size),
            '--base-epochs', str(epochs),
            '--hnm-rounds', '0',
            '--seed', str(seed),
            '--output-dir', str(model_dir),
        ]
        if balanced:
            train_argv.append('--balanced')

        sys.argv = train_argv
        try:
            metrics = train_main_fn()
        finally:
            sys.argv = original_argv

        elapsed = time.time() - start_time

        result_row.update({
            'best_epoch': metrics.get('best_epoch', ''),
            'best_val_auc': f"{metrics.get('best_auc', 0):.6f}",
            'best_val_acc': f"{metrics.get('best_acc', 0):.6f}",
            'final_val_loss': f"{metrics.get('final_val_loss', 0):.6f}",
            'final_val_acc': f"{metrics.get('final_val_acc', 0):.6f}",
            'final_val_auc': f"{metrics.get('final_val_auc', 0):.6f}",
            'final_val_f1': f"{metrics.get('final_val_f1', 0):.6f}",
            'elapsed_sec': f"{elapsed:.1f}",
            'status': 'ok',
        })

        logger.info(f"Finished LAYER_SIZE={layer_size}: best_auc={metrics.get('best_auc', 0):.4f}, best_acc={metrics.get('best_acc', 0):.4f}, elapsed={elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        result_row.update({
            'best_epoch': '',
            'best_val_auc': '',
            'best_val_acc': '',
            'final_val_loss': '',
            'final_val_acc': '',
            'final_val_auc': '',
            'final_val_f1': '',
            'elapsed_sec': f"{elapsed:.1f}",
            'status': f'error: {str(e)[:200]}',
        })
        logger.error(f"Failed LAYER_SIZE={layer_size}: {e}", exc_info=True)

    return result_row


def print_summary(csv_path: Path):
    if not csv_path.exists():
        logger.info("No completed results found.")
        return

    logger.info("Ablation Summary:")
    logger.info(f"{'Layer Size':>12} {'Best AUC':>12} {'Best Acc':>12} {'Final AUC':>12} {'Elapsed':>12} {'Status':>10}")

    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda r: int(r.get('layer_size', 0)))

    best_auc_overall = -1.0
    best_ls = None
    for row in rows:
        ls = row.get('layer_size', '?')
        bauc = row.get('best_val_auc', '')
        bacc = row.get('best_val_acc', '')
        fauc = row.get('final_val_auc', '')
        elapsed = row.get('elapsed_sec', '')
        status = row.get('status', '')[:8]

        logger.info(f"{ls:>12} {bauc:>12} {bacc:>12} {fauc:>12} {elapsed:>12} {status:>10}")

        try:
            auc_val = float(bauc)
            if auc_val > best_auc_overall:
                best_auc_overall = auc_val
                best_ls = ls
        except (ValueError, TypeError):
            pass

    if best_ls is not None:
        logger.info(f"Best LAYER_SIZE: {best_ls} (AUC = {best_auc_overall:.6f})")


def main():
    p = argparse.ArgumentParser(description='LAYER_SIZE 消融实验',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sizes', nargs='+', type=int, default=DEFAULT_SIZES,
                   help='待测试的 LAYER_SIZE 列表')
    p.add_argument('--batch-size', type=int, default=256,
                   help='训练 batch size')
    p.add_argument('--epochs', type=int, default=40,
                   help='每个配置的训练轮数')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--balanced', action='store_true',
                   help='启用类别平衡采样')
    p.add_argument('--csv-path', type=Path, default=CSV_PATH,
                   help='结果 CSV 输出路径')
    p.add_argument('--dry-run', action='store_true',
                   help='预览将要运行的配置')
    p.add_argument('--rerun-failed', action='store_true',
                   help='重新运行失败的配置')
    args = p.parse_args()

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)

    train_main_fn = load_train_main()

    completed = get_completed_sizes(args.csv_path)
    skip_sizes = completed if not args.rerun_failed else set()

    sizes_to_run = [s for s in args.sizes if s not in skip_sizes]

    logger.info(f"Target configurations: {args.sizes}")
    logger.info(f"Already completed: {sorted(completed)}")
    logger.info(f"To run now: {sizes_to_run}")

    if not sizes_to_run and not args.dry_run:
        logger.info("All configurations completed.")
        print_summary(args.csv_path)
        return

    total_start = time.time()
    for ls in sizes_to_run:
        row = run_single_ablation(
            train_main_fn=train_main_fn,
            layer_size=ls,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            balanced=args.balanced,
            dry_run=args.dry_run,
        )
        append_csv_row(args.csv_path, row)

    total_elapsed = time.time() - total_start
    if not args.dry_run:
        logger.info(f"Ablation complete. Total elapsed time: {total_elapsed:.1f}s")
        print_summary(args.csv_path)


if __name__ == '__main__':
    main()