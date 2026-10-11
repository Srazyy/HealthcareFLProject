"""
Script to transform, label (Option 1: Surgical vs. Medical), and pool
mtsamples.csv and PMC-patients.csv into unified clinical datasets.

Transformation rules:
  1. From mtsamples.csv:
     - Merges `description` and `transcription` into a single `claim` text column.
     - Maps records to:
         Label 1 = surgical_procedural (invasive/operative procedures)
         Label 0 = medical_conservative (pharmacological / non-surgical management)
     - Removes all other original columns.

  2. From PMC-patients.csv:
     - Merges `title`, `patient`, and `age` into a single `claim` text column.
     - Maps records to:
         Label 1 = surgical_procedural
         Label 0 = medical_conservative
     - Removes all other original columns.

Outputs generated in data/:
  - data/pooled_clinical_data_full.csv     (All processed records from both sources)
  - data/pooled_clinical_data_full.parquet (High-performance Parquet format of the full pool)
  - data/medical_claims_3k.parquet         (Balanced 10,000-record benchmark slice for FL training)
  - data/medical_claims_3k.csv             (CSV version of the 10,000-record benchmark slice)
  - data/pooled_data.csv                   (Convenience copy for manual inspection)
"""

import ast
import html
import logging
import re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Comprehensive surgical/procedural regex pattern
SURGICAL_PATTERN = re.compile(
    r"\b("
    r"surger(y|ies|ical)|operat(ion|ive|ed)|resect(ion|ed)?|excis(ion|ed)?|"
    r"biops(y|ies|ied)|laparoscop(y|ic)|thoracotom(y|ic)|laparotom(y|ic)|"
    r"craniotom(y|ic)|transplant(ation|ed)?|intubat(ion|ed)|angioplast(y|ic)|"
    r"stent(ing|ed)?|catheter(ization|ized)?|colectom(y|ic)|mastectom(y|ic)|"
    r"appendectom(y|ic)|cholecystectom(y|ic)|arthroscop(y|ic)|endoscop(y|ic)|"
    r"repair(ed)?|graft(ing|ed)?|debridement|prosthes(is|es)|implanta?t(ion|ed)?|"
    r"amputat(ion|ed)?|anastomos(is|es)|incis(ion|ed)|cannulat(ion|ed)|arthroplast(y|ic)"
    r")\b",
    re.IGNORECASE,
)

SURGICAL_SPECIALTIES = {
    "surgery",
    "orthopedic",
    "neurosurgery",
    "cosmetic / plastic surgery",
    "obstetrics / gynecology",
    "urology",
    "ophthalmology",
    "ent - otolaryngology",
    "podiatry",
    "dentistry",
    "bariatrics",
}


def clean_text(text: str) -> str:
    """Cleans raw text of HTML entities, excessive spaces, and noisy characters."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pmc_age(age_val) -> str:
    """Parses PMC age list string e.g. [[60.0, 'year']] into readable text."""
    if not age_val or pd.isna(age_val):
        return ""
    try:
        val_str = str(age_val).strip()
        if val_str.startswith("["):
            parsed = ast.literal_eval(val_str)
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                item = parsed[0]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    num, unit = item[0], item[1]
                    num_str = f"{int(num)}" if float(num).is_integer() else f"{num}"
                    return f"{num_str} {unit}s old"
    except Exception:
        pass
    return ""


def process_mtsamples(file_path: Path) -> pd.DataFrame:
    """Loads and formats mtsamples.csv keeping only merged text and Option 1 label."""
    logger.info("Loading mtsamples.csv from %s...", file_path)
    df = pd.read_csv(file_path)

    records = []
    for _, row in df.iterrows():
        desc = clean_text(str(row.get("description", "")))
        trans = clean_text(str(row.get("transcription", "")))

        # Merge description and transcription into a single text column
        if not trans:
            claim_text = desc
        elif not desc or desc.lower() in trans.lower()[: len(desc) + 40]:
            claim_text = trans
        else:
            claim_text = f"Clinical Summary: {desc}\n\nClinical Record: {trans}"

        if len(claim_text) < 30:
            continue

        # Option 1 Label: Surgical/Procedural (1) vs. Medical/Conservative (0)
        specialty = str(row.get("medical_specialty", "")).strip().lower()
        is_surgical_specialty = specialty in SURGICAL_SPECIALTIES
        has_surgical_keywords = bool(SURGICAL_PATTERN.search(claim_text))

        is_surgical = is_surgical_specialty or has_surgical_keywords
        label = 1 if is_surgical else 0
        label_name = "surgical_procedural" if is_surgical else "medical_conservative"

        records.append({
            "source": "mtsamples",
            "claim": claim_text,
            "label": label,
            "label_name": label_name,
        })

    result_df = pd.DataFrame(records)
    logger.info("Processed MTSamples: %d records (Surgical: %d, Medical: %d)",
                len(result_df),
                int((result_df["label"] == 1).sum()),
                int((result_df["label"] == 0).sum()))
    return result_df


def process_pmc_patients(file_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Loads and formats PMC-patients.csv merging title, patient, and age."""
    logger.info("Loading PMC-patients.csv from %s...", file_path)
    df = pd.read_csv(file_path, nrows=max_rows)

    records = []
    for _, row in df.iterrows():
        title = clean_text(str(row.get("title", "")))
        patient = clean_text(str(row.get("patient", "")))
        age_str = parse_pmc_age(row.get("age", ""))

        if len(patient) < 30:
            continue

        # Merge title, patient narrative, and age into a single text column
        header_parts = []
        if title:
            header_parts.append(f"Case Title: {title}")
        if age_str:
            header_parts.append(f"Age: {age_str}")

        if header_parts:
            claim_text = f"{' | '.join(header_parts)}\n\nPatient History & Clinical Presentation: {patient}"
        else:
            claim_text = patient

        # Option 1 Label: Surgical/Procedural (1) vs. Medical/Conservative (0)
        has_surgical = bool(SURGICAL_PATTERN.search(patient)) or bool(SURGICAL_PATTERN.search(title))
        label = 1 if has_surgical else 0
        label_name = "surgical_procedural" if has_surgical else "medical_conservative"

        records.append({
            "source": "pmc-patients",
            "claim": claim_text,
            "label": label,
            "label_name": label_name,
        })

    result_df = pd.DataFrame(records)
    logger.info("Processed PMC-Patients: %d records (Surgical: %d, Medical: %d)",
                len(result_df),
                int((result_df["label"] == 1).sum()),
                int((result_df["label"] == 0).sum()))
    return result_df


def create_balanced_subset(
    df_mtsamples: pd.DataFrame,
    df_pmc: pd.DataFrame,
    target_total: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Creates a balanced benchmark dataset: 50% MTSamples / 50% PMC-Patients, 50% Class 0 / 50% Class 1."""
    half = target_total // 2  # 5,000 per source
    quarter = half // 2       # 2,500 per class per source

    # Sample MTSamples
    mt_surg = df_mtsamples[df_mtsamples["label"] == 1]
    mt_med = df_mtsamples[df_mtsamples["label"] == 0]

    # MTSamples might have fewer medical rows, so balance appropriately
    n_mt_med = min(quarter, len(mt_med))
    n_mt_surg = min(half - n_mt_med, len(mt_surg))

    mt_sampled_med = mt_med.sample(n=n_mt_med, random_state=seed)
    mt_sampled_surg = mt_surg.sample(n=n_mt_surg, random_state=seed)
    mt_subset = pd.concat([mt_sampled_med, mt_sampled_surg], ignore_index=True)

    # Balance remaining from PMC-Patients to ensure exact 50/50 overall class split
    remaining_total = target_total - len(mt_subset)
    target_class0_total = target_total // 2
    needed_class0 = target_class0_total - int((mt_subset["label"] == 0).sum())
    needed_class1 = remaining_total - needed_class0

    pmc_surg = df_pmc[df_pmc["label"] == 1]
    pmc_med = df_pmc[df_pmc["label"] == 0]

    pmc_sampled_med = pmc_med.sample(n=needed_class0, random_state=seed)
    pmc_sampled_surg = pmc_surg.sample(n=needed_class1, random_state=seed)
    pmc_subset = pd.concat([pmc_sampled_med, pmc_sampled_surg], ignore_index=True)

    combined = pd.concat([mt_subset, pmc_subset], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    combined.insert(0, "claim_id", [f"CLIN-{i+1:05d}" for i in range(len(combined))])
    return combined


def main():
    mtsamples_path = DATA_DIR / "mtsamples.csv"
    pmc_path = DATA_DIR / "PMC-patients.csv"

    if not mtsamples_path.exists() or not pmc_path.exists():
        logger.error("Required CSV files not found in %s", DATA_DIR)
        return

    # 1. Process both datasets
    df_mtsamples = process_mtsamples(mtsamples_path)
    df_pmc = process_pmc_patients(pmc_path)

    # 2. Build full combined pool
    logger.info("Combining into full master dataset...")
    full_combined = pd.concat([df_mtsamples, df_pmc], ignore_index=True)
    full_combined = full_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full_combined.insert(0, "claim_id", [f"MASTER-{i+1:06d}" for i in range(len(full_combined))])

    # Save Full Pool
    full_csv = DATA_DIR / "pooled_clinical_data_full.csv"
    full_parquet = DATA_DIR / "pooled_clinical_data_full.parquet"
    full_combined.to_csv(full_csv, index=False)
    full_combined.to_parquet(full_parquet, index=False)
    logger.info("Saved FULL pool: %s (%d records)", full_csv, len(full_combined))
    logger.info("Saved FULL pool: %s", full_parquet)

    # 3. Build balanced 10,000-record benchmark slice for Federated Learning
    logger.info("Creating balanced 10,000-record benchmark dataset for Federated Learning...")
    benchmark_df = create_balanced_subset(df_mtsamples, df_pmc, target_total=10000, seed=42)

    bench_parquet = DATA_DIR / "medical_claims_3k.parquet"
    bench_csv = DATA_DIR / "medical_claims_3k.csv"
    review_csv = DATA_DIR / "pooled_data.csv"

    benchmark_df.to_parquet(bench_parquet, index=False)
    benchmark_df.to_csv(bench_csv, index=False)
    benchmark_df.to_csv(review_csv, index=False)

    logger.info("Saved FL Benchmark Parquet: %s (%d records)", bench_parquet, len(benchmark_df))
    logger.info("Saved FL Benchmark CSV:     %s", bench_csv)
    logger.info("Saved Review CSV:           %s", review_csv)

    # Summary
    logger.info("=" * 60)
    logger.info("  TRANSFORMATION & POOLING COMPLETE")
    logger.info("=" * 60)
    logger.info("Full Dataset Pool: %d records", len(full_combined))
    logger.info("  - MTSamples:    %d", len(df_mtsamples))
    logger.info("  - PMC-Patients: %d", len(df_pmc))
    logger.info("Benchmark Slice (10,000 records) Distribution:")
    for lbl, count in benchmark_df["label"].value_counts().items():
        name = "surgical_procedural" if lbl == 1 else "medical_conservative"
        pct = 100.0 * count / len(benchmark_df)
        logger.info("  - Label %d (%s): %d records (%.1f%%)", lbl, name, count, pct)
    logger.info("Source breakdown in Benchmark Slice:")
    for src, count in benchmark_df["source"].value_counts().items():
        pct = 100.0 * count / len(benchmark_df)
        logger.info("  - %s: %d records (%.1f%%)", src, count, pct)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
