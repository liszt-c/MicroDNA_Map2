"""
src/pipeline/cnvkit_pipeline.py - CNVkit 分析流程封装 (同步修复安全提取与清理逻辑)
"""
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import pysam

from config import (
    BOWTIE2,
    BOWTIE2_BUILD,
    CNVKIT,
    CNVKIT_REF_CNN,
    CNVKIT_TEMP_DIR,
    HG19_FA,
    SAMTOOLS,
)
from ..utils import (
    ensure_bam_index,
    ensure_faidx,
    is_bam_valid,
    resolve_bowtie2_index,
    run_command,
    setup_logger,
)

logger = setup_logger('cnvkit_pipeline')

CALL_CNS_REQUIRED_COLS = ('chromosome', 'start', 'end', 'log2')


class CNVKitPipeline:
    def __init__(self, output_dir: Optional[Path] = None, ref_genome: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else CNVKIT_TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ref_genome = Path(ref_genome) if ref_genome else HG19_FA
        if not self.ref_genome.exists():
            raise FileNotFoundError(
                f"Reference genome not found: {self.ref_genome}\n"
                f"Please place hg19.fa into {self.ref_genome.parent}/"
            )
        prefix, _ = resolve_bowtie2_index(self.ref_genome)
        self.index_prefix = prefix
        self._index_checked = False

        ensure_faidx(self.ref_genome, SAMTOOLS)

    def build_index(self):
        """构建 Bowtie2 索引 (若缺失)"""
        prefix, exists = resolve_bowtie2_index(self.ref_genome)
        self.index_prefix = prefix
        if exists:
            logger.debug(f"Bowtie2 index already exists at {self.index_prefix}, skip building.")
            self._index_checked = True
            return

        logger.info(f"Building Bowtie2 index: {self.ref_genome} -> {self.index_prefix}")
        run_command([BOWTIE2_BUILD, "-f", str(self.ref_genome), str(self.index_prefix)])
        self._index_checked = True

    def align(
        self,
        fastq1: Path,
        fastq2: Path,
        sample_name: str,
        threads: int = 8,
        force_align: bool = False,
    ) -> Path:
        """FASTQ -> sorted & indexed BAM (支持复用已存在的有效 BAM 与索引)"""
        fastq1, fastq2 = Path(fastq1), Path(fastq2)
        bam_file = self.output_dir / f"{sample_name}.bam"

        if not force_align and is_bam_valid(bam_file, SAMTOOLS):
            logger.info(f"[{sample_name}] Found existing valid BAM: {bam_file.name}. Skipping alignment.")
            ensure_bam_index(bam_file, SAMTOOLS)
            return bam_file

        for fq in (fastq1, fastq2):
            if not fq.exists():
                raise FileNotFoundError(f"FASTQ not found: {fq}")

        if not self._index_checked:
            self.build_index()

        sam_file = self.output_dir / f"{sample_name}.sam"

        logger.info(f"[{sample_name}] bowtie2 aligning ({threads} threads) ...")
        run_command([
            BOWTIE2, "-p", str(threads),
            "-x", str(self.index_prefix),
            "-1", str(fastq1), "-2", str(fastq2),
            "-S", str(sam_file),
        ])

        logger.info(f"[{sample_name}] samtools sort -> {bam_file.name} ...")
        run_command([SAMTOOLS, "sort", f"-@{threads}", "-o", str(bam_file), str(sam_file)])
        try:
            sam_file.unlink()
        except OSError as e:
            logger.warning(f"Failed to remove {sam_file}: {e}")

        logger.info(f"[{sample_name}] Ensuring BAM index for {bam_file.name} ...")
        ensure_bam_index(bam_file, SAMTOOLS)
        logger.info(f"[{sample_name}] BAM ready: {bam_file}")
        return bam_file

    def call_cnv(self, bam_file: Path, sample_name: str, threads: int = 8) -> Path:
        bam_file = Path(bam_file)
        ensure_bam_index(bam_file, SAMTOOLS)

        batch_cmd = [
            CNVKIT, "batch", "-m", "wgs", "-p", str(threads),
            "-d", str(self.output_dir)
        ]
        if CNVKIT_REF_CNN.exists():
            batch_cmd += ["-r", str(CNVKIT_REF_CNN)]
            logger.info(f"[{sample_name}] Using CNVkit reference profile: {CNVKIT_REF_CNN.name}")
        else:
            logger.info(f"[{sample_name}] No reference .cnn found, CNVkit will build a flat reference.")
        batch_cmd.append(str(bam_file))

        logger.info(f"[{sample_name}] cnvkit batch ...")
        run_command(batch_cmd)

        cns_file = self.output_dir / f"{sample_name}.cns"
        if not cns_file.exists():
            raise FileNotFoundError(f"cnvkit batch did not produce {cns_file}. See stderr above.")

        call_out = self.output_dir / f"{sample_name}.call.cns"
        logger.info(f"[{sample_name}] cnvkit call ...")
        run_command([CNVKIT, "call", str(cns_file), "-o", str(call_out)])

        if not call_out.exists():
            raise FileNotFoundError(f"cnvkit call did not produce {call_out}")
        return call_out

    def align_and_call(
        self,
        fastq1,
        fastq2,
        sample_name: str,
        threads: int = 8,
        force_align: bool = False,
    ) -> Path:
        bam = self.align(fastq1, fastq2, sample_name, threads=threads, force_align=force_align)
        return self.call_cnv(bam, sample_name, threads=threads)

    def extract_candidates(
        self,
        call_cns: Path,
        min_log2: float = None,
        min_size: int = 0,
        max_size: int = None,
        sample_name: str = None
    ) -> list:
        call_cns = Path(call_cns)
        if not call_cns.exists():
            raise FileNotFoundError(f"CNV call file not found: {call_cns}")

        df = pd.read_csv(call_cns, sep='\t', comment=None, dtype=str)
        missing = [c for c in CALL_CNS_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{call_cns.name} lacks required columns {missing}; found {list(df.columns)}")

        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')
        df['log2'] = pd.to_numeric(df['log2'], errors='coerce')
        df = df.dropna(subset=['start', 'end'])
        df['size'] = df['end'] - df['start']

        n0 = len(df)
        if min_log2 is not None:
            df = df[df['log2'] >= float(min_log2)]
        if min_size:
            df = df[df['size'] >= int(min_size)]
        if max_size:
            df = df[df['size'] <= int(max_size)]
        logger.info(
            f"[{call_cns.name}] segments: {n0} total -> {len(df)} after filtering "
            f"(min_log2={min_log2}, min_size={min_size}, max_size={max_size})"
        )

        prefix = f"{sample_name}_" if sample_name else ""
        candidates = []
        for i, (_, row) in enumerate(df.iterrows()):
            chrom = str(row['chromosome']).strip()
            start, end = int(row['start']), int(row['end'])
            candidates.append({
                'name': f"{prefix}cnv{i}",
                'chrom': chrom,
                'start': start,
                'end': end,
                'log2': float(row['log2']) if pd.notna(row['log2']) else float('nan'),
                'size': end - start,
            })
        return candidates

    def extract_sequences(self, candidates: List[Dict], output_fa: Path) -> int:
        """基于 pysam.FastaFile 高性能安全提取序列"""
        output_fa = Path(output_fa)
        if not candidates:
            logger.warning("No candidates to extract; writing empty FASTA.")
            output_fa.write_text("")
            return 0

        ensure_faidx(self.ref_genome, SAMTOOLS)

        uniq, seen = [], set()
        for c in candidates:
            key = (c['chrom'], c['start'], c['end'])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        written = 0
        with pysam.FastaFile(str(self.ref_genome)) as fa, open(output_fa, 'w', encoding='utf-8') as f:
            ref_chroms = set(fa.references)
            chrom_map = {}
            for chrom in ref_chroms:
                chrom_map[chrom] = chrom
                bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
                chrom_map[bare] = chrom
                chrom_map[f"chr{bare}"] = chrom
                chrom_map[chrom.upper()] = chrom
                chrom_map[chrom.lower()] = chrom

            for c in uniq:
                raw_chrom = str(c['chrom']).strip()
                ref_chrom = chrom_map.get(raw_chrom)
                if not ref_chrom:
                    continue

                chrom_len = fa.get_reference_length(ref_chrom)
                start = max(1, int(c['start']))
                end = min(int(c['end']), chrom_len)
                if start > end:
                    continue

                try:
                    seq = fa.fetch(ref_chrom, start - 1, end).upper()
                except Exception as e:
                    continue

                if not seq:
                    continue

                f.write(f">{c['name']}|{ref_chrom}:{start}-{end}\n")
                for i in range(0, len(seq), 70):
                    f.write(seq[i : i + 70] + "\n")
                written += 1

        logger.info(f"Wrote {written}/{len(candidates)} sequences to {output_fa.name}")
        return written

    @staticmethod
    def candidates_to_bed_rows(candidates: list) -> list:
        rows = []
        for c in candidates:
            rows.append((c['chrom'], max(0, int(c['start']) - 1), int(c['end'])))
        return rows

    def cleanup_sample(self, sample_name: str, keep_bam: bool = False):
        """删除单个样本的临时文件 (保留 .call.cns 与 .cns)"""
        patterns = [
            f"{sample_name}.sam", f"{sample_name}.cnr",
            f"{sample_name}_candidates.fa",
            f"{sample_name}.antitargetcoverage.cnn", f"{sample_name}.targetcoverage.cnn"
        ]
        if not keep_bam:
            patterns += [
                f"{sample_name}.bam",
                f"{sample_name}.bam.bai",
                f"{sample_name}.bai",
            ]
        removed = 0
        for pat in patterns:
            for p in self.output_dir.glob(pat):
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning(f"Failed to remove {p}: {e}")
        logger.info(f"[{sample_name}] Removed {removed} intermediate files.")
        return removed