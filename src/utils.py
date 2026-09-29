"""
src/utils.py - 通用工具函数 (增强 pysam 原生极速安全提取)
"""
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple


def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    """设置日志记录器 (防止重复添加 handler)"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def run_command(cmd: list, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """安全地执行子进程命令 (列表形式, 无 shell 注入风险)"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None
        )
        return result
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[ERROR] Command failed: {' '.join(map(str, cmd))}\n")
        sys.stderr.write(f"[ERROR] returncode={e.returncode}\n")
        if e.stdout:
            sys.stderr.write(f"[ERROR] stdout: {e.stdout[-2000:]}\n")
        if e.stderr:
            sys.stderr.write(f"[ERROR] stderr: {e.stderr[-2000:]}\n")
        raise
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Executable not found: '{cmd[0]}'. "
            f"Please install it or set the full path in config.py"
        )


def parse_fasta_text(text: str):
    """解析 FASTA 格式文本, 返回 [(header_line, sequence), ...]"""
    records = []
    header = None
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(parts)))
            header = line
            parts = []
        else:
            parts.append(line)
    if header is not None:
        records.append((header, ''.join(parts)))
    return records


def parse_fasta_file(path: Path):
    """解析 FASTA 文件 (整读入内存, 仅适用于小文件)"""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return parse_fasta_text(f.read())


def iter_fasta_file(path: Path) -> Iterator[Tuple[str, str]]:
    """惰性解析 FASTA 文件, 逐条 yield (header_line, sequence)"""
    path = Path(path)
    header = None
    parts = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(parts)
                header = line
                parts = []
            else:
                if header is not None:
                    parts.append(line)
    if header is not None:
        yield header, ''.join(parts)


def read_fai(ref_fa: Path) -> dict:
    """读取 .fai 索引, 返回 {'lengths': {chrom: length}, 'order': {chrom: index}}"""
    fai_path = Path(str(ref_fa) + '.fai')
    lengths, order = {}, {}
    if fai_path.exists():
        with open(fai_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2:
                    lengths[parts[0]] = int(parts[1])
                    order[parts[0]] = i
    return {'lengths': lengths, 'order': order}


def ensure_faidx(ref_fa: Path, samtools_bin: str = 'samtools') -> None:
    """确保参考基因组已建立 samtools faidx 索引"""
    ref_fa = Path(ref_fa)
    if not ref_fa.exists():
        raise FileNotFoundError(f"Reference genome not found: {ref_fa}")
    fai_path = Path(str(ref_fa) + '.fai')
    if not fai_path.exists():
        run_command([samtools_bin, 'faidx', str(ref_fa)])


def is_bam_valid(bam_path: Path, samtools_bin: str = 'samtools') -> bool:
    """快速检查 BAM 文件是否存在且完整未损坏"""
    bam_path = Path(bam_path)
    if not bam_path.exists() or bam_path.stat().st_size == 0:
        return False

    try:
        res = subprocess.run(
            [samtools_bin, "quickcheck", str(bam_path)],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            return True
        return False
    except Exception:
        pass

    try:
        import pysam
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            if not bam.header or not bam.references:
                return False
        return True
    except Exception:
        return False


def ensure_bam_index(bam_path: Path, samtools_bin: str = 'samtools') -> Path:
    """确保 BAM 文件已建立有效的 .bai 索引，避免重复生成索引"""
    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    bai1 = Path(str(bam_path) + ".bai")
    bai2 = bam_path.with_suffix(".bai")

    bam_mtime = bam_path.stat().st_mtime
    for bai in [bai1, bai2]:
        if bai.exists() and bai.stat().st_size > 0 and bai.stat().st_mtime >= bam_mtime:
            return bai

    run_command([samtools_bin, "index", str(bam_path)])
    return bai1 if bai1.exists() else bai2


def resolve_bowtie2_index(ref_genome: Path) -> Tuple[Path, bool]:
    """检查参考基因组对应的 Bowtie2 索引是否存在"""
    ref_genome = Path(ref_genome)
    parent = ref_genome.parent
    prefixes = [parent / ref_genome.stem, parent / ref_genome.name]

    for prefix in prefixes:
        prefix_str = str(prefix)
        has_small = all(
            Path(f"{prefix_str}{ext}").exists() and Path(f"{prefix_str}{ext}").stat().st_size > 0
            for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
        )
        has_large = all(
            Path(f"{prefix_str}{ext}").exists() and Path(f"{prefix_str}{ext}").stat().st_size > 0
            for ext in [".1.bt2l", ".2.bt2l", ".3.bt2l", ".4.bt2l", ".rev.1.bt2l", ".rev.2.bt2l"]
        )
        if has_small or has_large:
            return prefix, True

    return prefixes[0], False


def extract_regions(ref_fa: Path, regions: list, samtools_bin: str = 'samtools'):
    """
    批量提取基因组区域序列 (优先使用 pysam.FastaFile 极速原生提取, 无管道截断风险)
    """
    ref_fa = Path(ref_fa)
    ensure_faidx(ref_fa, samtools_bin)
    fai = read_fai(ref_fa)
    lengths, order = fai['lengths'], fai['order']
    if not lengths:
        raise FileNotFoundError(f".fai index missing for {ref_fa}, run ensure_faidx first")

    # 染色体别名自动解析器 (兼容带/不带 chr 前缀及大小写)
    known = set(lengths)
    chrom_cache = {}

    def resolve_chrom(chrom_str: str):
        c = str(chrom_str).strip()
        if not c or c.lower() in ('nan', 'none'):
            return None
        if c in chrom_cache:
            return chrom_cache[c]
        bare = c[3:] if c.lower().startswith('chr') else c
        candidates = [c, f"chr{bare}", bare, c.upper(), c.lower(),
                      f"chr{bare.upper()}", f"chr{bare.lower()}"]
        hit = next((cand for cand in dict.fromkeys(candidates) if cand in known), None)
        chrom_cache[c] = hit
        return hit

    valid = []
    skipped = []
    for region in regions:
        chrom, start, end = region
        target_chrom = resolve_chrom(chrom)
        if target_chrom is None:
            skipped.append((region, f"chrom '{chrom}' not in reference index"))
            continue
        max_len = lengths[target_chrom]
        s = max(1, int(start))
        e = min(int(end), max_len)
        if s > e:
            skipped.append((region, f"invalid coordinate range after clamping ({s}>{e})"))
            continue
        valid.append((region, target_chrom, s, e))

    if not valid:
        return {}, skipped

    # 优先采用 pysam 原生直接寻道提取
    try:
        import pysam
        seq_map = {}
        with pysam.FastaFile(str(ref_fa)) as fa:
            for orig_key, c, s, e in valid:
                try:
                    # 1-based 闭区间 [s, e] 对应 pysam 0-based half-open [s - 1, e)
                    seq = fa.fetch(c, s - 1, e)
                    seq_map[orig_key] = seq
                except Exception as err:
                    skipped.append((orig_key, f"pysam error: {err}"))
        return seq_map, skipped
    except ImportError:
        pass

    # 备选回退方案: 调用外部 samtools 命令 (添加 -c 参数防止单点错误中断全流程)
    valid.sort(key=lambda t: (order[t[1]], t[2]))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as tf:
        for _, chrom, start, end in valid:
            tf.write(f"{chrom}:{start}-{end}\n")
        tf_path = tf.name

    try:
        result = run_command([samtools_bin, 'faidx', '-c', str(ref_fa), '-r', tf_path])
    finally:
        os.unlink(tf_path)

    records = parse_fasta_text(result.stdout)
    seq_map = {}
    for (orig_key, _c, _s, _e), (_hdr, seq) in zip(valid, records):
        seq_map[orig_key] = seq

    return seq_map, skipped