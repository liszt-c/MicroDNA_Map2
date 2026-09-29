"""
scripts/batch_process.py - 从 FASTQ 到 MicroDNA 鉴定的全流程

特性:
  - 智能参数比对机制：自动解析已有 .call.cns 中的参数。若修改了 --fold-change 等参数，自动感知并重新扫描；若参数未变，秒级复用。
  - 支持 --force-call 显式强制重跑初筛。
  - 使用流式进度条实时展示深度学习分类进度。
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from tqdm import tqdm

from config import (
    CNVKIT_TEMP_DIR,
    DEFAULT_MICRO_CLUSTER_MAX_LEN,
    DEFAULT_MICRO_FOLD_CHANGE,
    DEFAULT_MICRO_MAX_LEN,
    DEFAULT_MICRO_MIN_LEN,
    DEFAULT_MICRO_WINDOW_SIZE,
    MICRO_COVERAGE_TEMP_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    RAW_DATA_DIR,
)
from src.pipeline import get_pipeline
from src.utils import iter_fasta_file, setup_logger

_predict_module = None


def _get_predict_funcs():
    global _predict_module
    if _predict_module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "predict", str(Path(__file__).resolve().parent / "predict.py")
        )
        _predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_predict_module)
    return _predict_module.load_model, _predict_module.predict_long_single


logger = setup_logger("batch_process")


def find_fastq_pairs(directory: Path):
    """查找配对的 fastq 文件 (支持 _1/_2, R1/R2 命名)"""
    pairs = []
    seen_bases = set()

    for pattern in ["*_1.fastq", "*_1.fq", "*_R1.fastq", "*_R1.fq"]:
        for r1 in sorted(directory.glob(pattern)):
            name = r1.name
            for suffix in ["_1.fastq", "_1.fq", "_R1.fastq", "_R1.fq"]:
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    break
            else:
                continue

            if base in seen_bases:
                continue
            seen_bases.add(base)

            r2_candidates = [
                directory / f"{base}_2.fastq",
                directory / f"{base}_2.fq",
                directory / f"{base}_R2.fastq",
                directory / f"{base}_R2.fq",
            ]
            r2 = next((c for c in r2_candidates if c.exists()), None)
            if r2:
                pairs.append((base, r1, r2))
            else:
                logger.warning(f"Missing R2 pair for {r1.name}")

    return pairs


def parse_call_cns_params(call_cns_file: Path) -> Dict[str, str]:
    """读取 .call.cns 首行的参数指纹字典"""
    params = {}
    try:
        with open(call_cns_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.startswith("# params:"):
                raw_str = first_line.replace("# params:", "").strip()
                for item in raw_str.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        params[k.strip()] = v.strip()
    except Exception:
        pass
    return params


def can_reuse_call_cns(call_cns_file: Path, expected_params: Dict[str, str]) -> Tuple[bool, str]:
    """比对已有 call.cns 与当前参数是否严格一致"""
    if not call_cns_file.exists() or call_cns_file.stat().st_size == 0:
        return False, "File does not exist or is empty"

    cached_params = parse_call_cns_params(call_cns_file)
    if not cached_params:
        return False, "No parameter fingerprint found in existing file"

    for k, v in expected_params.items():
        if k not in cached_params:
            return False, f"Missing parameter '{k}' in cached file"
        if cached_params[k] != v:
            return False, f"Parameter mismatch: '{k}' (cached={cached_params[k]} vs current={v})"

    return True, "Parameters perfectly match"


def main():
    p = argparse.ArgumentParser(
        description="Batch process FASTQ -> MicroDNA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=str(RAW_DATA_DIR), help="FASTQ 所在目录")
    p.add_argument(
        "--pipeline",
        choices=["micro_coverage", "cnvkit"],
        default="micro_coverage",
        help="候选区域初筛流程: 'micro_coverage' (自研微尺度预扫描) 或 'cnvkit' (传统大尺度 CNV)",
    )
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--limit", type=float, default=0.75, help="预测判定阈值")
    p.add_argument("--min-log2", type=float, default=None, help="CNV/覆盖度 log2 过滤阈值")
    p.add_argument("--min-cnv-size", type=int, default=0, help="最小候选片段长度 (bp)")
    p.add_argument("--max-cnv-size", type=int, default=None, help="最大候选片段长度 (bp)")
    p.add_argument("--batch-size", type=int, default=256, help="滑窗推理批次大小")
    p.add_argument("--min-region-len", type=int, default=150, help="最终预测最小区域长度")
    p.add_argument("--cleanup", action="store_true", help="完成后删除中间临时比对与候选文件 (安全保留 call.cns)")
    p.add_argument("--keep-bam", action="store_true", help="cleanup 时保留排序后的 BAM")
    p.add_argument("--force-align", action="store_true", help="即使已存在有效 BAM，也强制重新执行比对与排序")
    p.add_argument("--force-call", action="store_true", help="强制重新执行覆盖度初筛计算，忽略参数比对")
    p.add_argument("--model", default=None, help="深度学习模型权重路径")
    p.add_argument("--output-dir", default=str(PREDICTIONS_DIR), help="结果输出目录")

    # 微尺度局部覆盖度分析专属参数
    p.add_argument("--window-size", type=int, default=DEFAULT_MICRO_WINDOW_SIZE, help="微尺度分析窗口大小 (bp)")
    p.add_argument("--fold-change", type=float, default=DEFAULT_MICRO_FOLD_CHANGE, help="微尺度分析倍数阈值")
    p.add_argument("--cluster-max-len", type=int, default=DEFAULT_MICRO_CLUSTER_MAX_LEN, help="微环簇最大容忍上限 (bp)")

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_fastq_pairs(input_dir)
    if not pairs:
        logger.error(f"No paired FASTQ files found in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(pairs)} sample pair(s). Active pipeline: '{args.pipeline}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model) if args.model else MODEL_DIR / "best_model.pth"
    load_model_fn, predict_long_fn = _get_predict_funcs()
    model, ecc_class_index = load_model_fn(model_path, device)

    temp_dir = MICRO_COVERAGE_TEMP_DIR if args.pipeline == "micro_coverage" else CNVKIT_TEMP_DIR
    pipeline_kwargs = {}
    if args.pipeline == "micro_coverage":
        pipeline_kwargs = {
            "window_size": args.window_size,
            "fold_change": args.fold_change,
            "min_region_len": DEFAULT_MICRO_MIN_LEN,
            "max_region_len": DEFAULT_MICRO_MAX_LEN,
            "cluster_max_len": args.cluster_max_len,
        }

    pipeline = get_pipeline(args.pipeline, output_dir=temp_dir, **pipeline_kwargs)

    for base, r1, r2 in pairs:
        logger.info(f"\n{'='*60}\nProcessing sample: {base} [Pipeline: {args.pipeline}]\n{'='*60}")
        try:
            call_cns_file = temp_dir / f"{base}.call.cns"
            should_run_call = True

            if not args.force_call and args.pipeline == "micro_coverage":
                expected_params = pipeline.get_parameter_dict()
                can_reuse, reason = can_reuse_call_cns(call_cns_file, expected_params)
                if can_reuse:
                    logger.info(f"[{base}] Reusing existing '{call_cns_file.name}' (Parameters match current setting).")
                    call_cns = call_cns_file
                    should_run_call = False
                else:
                    logger.info(f"[{base}] Will re-run coverage profiling. Reason: {reason}.")

            if should_run_call:
                call_cns = pipeline.align_and_call(
                    r1, r2, base, threads=args.threads, force_align=args.force_align
                )

            candidates = pipeline.extract_candidates(
                call_cns,
                min_log2=args.min_log2,
                min_size=args.min_cnv_size,
                max_size=args.max_cnv_size,
                sample_name=base,
            )

            if not candidates:
                logger.info(f"No candidates prioritized for {base}, skipping downstream inference.")
                if args.cleanup:
                    pipeline.cleanup_sample(base, keep_bam=args.keep_bam)
                continue

            logger.info(f"{len(candidates)} candidate region(s) prioritized for {base}")

            temp_fa = temp_dir / f"{base}_candidates.fa"
            n_written = pipeline.extract_sequences(candidates, temp_fa)
            if n_written == 0:
                logger.warning(f"No sequences extracted for {base}")
                if args.cleanup:
                    pipeline.cleanup_sample(base, keep_bam=args.keep_bam)
                continue

            passed_bed = []
            passed_fasta = []

            pbar = tqdm(
                iter_fasta_file(temp_fa),
                total=n_written,
                desc=f"Evaluating candidates ({base})",
                unit="region",
            )
            for rec_header, rec_seq in pbar:
                regions, seq_clean, chrom_info = predict_long_fn(
                    rec_header,
                    rec_seq,
                    model,
                    device,
                    ecc_class_index,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    min_region_len=args.min_region_len,
                )

                chrom = chrom_info.get("chrom", "")
                if chrom_info.get("has_position", False):
                    base_offset = max(0, chrom_info.get("start", 1) - 1)
                else:
                    base_offset = 0

                for rs, re in regions:
                    abs_s = base_offset + rs
                    abs_e = base_offset + re
                    passed_bed.append((chrom, abs_s, abs_e))
                    passed_fasta.append(
                        (f">{chrom}:{abs_s}-{abs_e}|prob>=limit", seq_clean[rs:re])
                    )
                pbar.set_postfix({"identified": len(passed_bed)})

            final_bed = output_dir / f"{base}_microDNA.bed"
            final_fa = output_dir / f"{base}_microDNA.fasta"

            with open(final_bed, "w", encoding="utf-8") as bf:
                for chrom, s, e in passed_bed:
                    bf.write(f"{chrom}\t{s}\t{e}\n")

            with open(final_fa, "w", encoding="utf-8") as ff:
                for hdr, seq in passed_fasta:
                    ff.write(hdr + "\n")
                    for i in range(0, len(seq), 70):
                        ff.write(seq[i : i + 70] + "\n")

            logger.info(f"Sample {base}: {len(passed_bed)} MicroDNA(s) identified.")
            logger.info(f"  BED  -> {final_bed}")
            logger.info(f"  FASTA-> {final_fa}")

            if args.cleanup:
                pipeline.cleanup_sample(base, keep_bam=args.keep_bam)

        except Exception as e:
            logger.error(f"Failed processing {base}: {e}", exc_info=True)

    logger.info("\nAll samples processed.")


if __name__ == "__main__":
    main()