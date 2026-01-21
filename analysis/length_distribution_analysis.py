import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============== CONFIG ==============
# File dùng để lấy max_len và tạo bins
BIN_SOURCE_PATH = "/home/anhkhoa/PhonoASR/result_new/result-tasa-w2i-lsvsc.json"

# Các file cần tính avg WER/CER theo bins
RESULT_PATHS = {
    "tasa-c2i-lsvsc": "/home/anhkhoa/PhonoASR/result_new/reevaluated-result-tasa-c2i-lsvsc.json",
    "tasa-w2i-lsvsc": "/home/anhkhoa/PhonoASR/result_new/result-tasa-w2i-lsvsc.json",
    "tasa-p2i-lsvsc": "/home/anhkhoa/PhonoASR/result_new/result-tasa-p2i-lsvsc.json",
}

# Output paths (tuỳ chỉnh)
OUT_PNG = "target_length_distribution.png"
OUT_DIST_CSV = "length_distribution_bins.csv"
OUT_SUMMARY_CSV = "wer_cer_by_length_bins.csv"
OUT_XLSX = "wer_cer_by_bins_pivots.xlsx"
# ====================================


def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data


def sent_len_words(s) -> int:
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    return len(s.split())


def build_equal_width_bins(max_len: int, n_bins: int = 5):
    """
    Chia [0..max_len] thành n_bins khoảng (equal-width).
    Trả về edges (len = n_bins+1) và labels (len = n_bins).
    """
    if max_len <= 0:
        edges = np.array([0, 1, 2, 3, 4, 5])  # fallback
        labels = [f"{edges[i]}–{edges[i+1]}" for i in range(5)]
        return edges, labels

    edges = np.linspace(0, max_len, n_bins + 1)

    # Convert to int edges and ensure strictly increasing
    edges_int = np.floor(edges).astype(int)
    for i in range(1, len(edges_int)):
        if edges_int[i] <= edges_int[i - 1]:
            edges_int[i] = edges_int[i - 1] + 1
    edges_int[-1] = max_len

    labels = [f"{edges_int[i]}–{edges_int[i+1]}" for i in range(n_bins)]
    return edges_int, labels


def assign_bin(lengths: np.ndarray, edges: np.ndarray, labels: list[str]):
    # right-inclusive; include_lowest=True để 0 vào bin đầu
    return pd.cut(
        lengths,
        bins=edges,
        include_lowest=True,
        right=True,
        labels=labels
    )


def length_distribution(records, edges, labels):
    lengths = np.array([sent_len_words(r.get("gold", "")) for r in records], dtype=int)
    bins = assign_bin(lengths, edges, labels)
    dist = (
        pd.Series(bins)
        .value_counts()
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    dist.columns = ["length_bin(words)", "count"]
    dist["pct"] = dist["count"] / dist["count"].sum() * 100
    return dist, lengths


def summarize_by_bins(records, edges, labels, file_key):
    lengths = np.array([sent_len_words(r.get("gold", "")) for r in records], dtype=int)
    bins = assign_bin(lengths, edges, labels)

    df = pd.DataFrame({
        "len_words": lengths,
        "bin": bins,
        "WER": [float(r.get("WER", np.nan)) for r in records],
        "CER": [float(r.get("CER", np.nan)) for r in records],
    })

    overall = pd.DataFrame([{
        "file": file_key,
        "bin": "OVERALL",
        "n": int(df.shape[0]),
        "avg_WER": float(df["WER"].mean()),
        "avg_CER": float(df["CER"].mean()),
        "avg_len_words": float(df["len_words"].mean()),
    }])

    per_bin = (
        df.groupby("bin", observed=False)
          .agg(
              n=("WER", "size"),
              avg_WER=("WER", "mean"),
              avg_CER=("CER", "mean"),
              avg_len_words=("len_words", "mean")
          )
          .reindex(labels)
          .reset_index()
    )
    per_bin.insert(0, "file", file_key)

    return pd.concat([overall, per_bin], ignore_index=True)


def main():
    # 1) Load bin source, compute max_len, build bins
    bin_source = load_json_list(BIN_SOURCE_PATH)
    lens_source = np.array([sent_len_words(r.get("gold", "")) for r in bin_source], dtype=int)
    max_len = int(lens_source.max()) if len(lens_source) else 0

    edges, labels = build_equal_width_bins(max_len=max_len, n_bins=5)
    print(f"[BIN SOURCE] {BIN_SOURCE_PATH}")
    print(f"max_len(words) = {max_len}")
    print(f"bins = {labels}")

    # 2) Distribution + plot (from bin source)
    dist_df, _ = length_distribution(bin_source, edges, labels)
    dist_df.to_csv(OUT_DIST_CSV, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    plt.bar(dist_df["length_bin(words)"], dist_df["count"])
    plt.xlabel("Target length bin (words)")
    plt.ylabel("Number of samples")
    plt.title(f"Target length distribution (source={BIN_SOURCE_PATH}), max_len={max_len}")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved dist csv: {OUT_DIST_CSV}")

    # 3) Summaries for each result file
    all_summaries = []
    for key, path in RESULT_PATHS.items():
        recs = load_json_list(path)
        summary = summarize_by_bins(recs, edges, labels, key)
        all_summaries.append(summary)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved summary csv: {OUT_SUMMARY_CSV}")

    # 4) Pivot sheets (optional)
    perbin = summary_df[summary_df["bin"] != "OVERALL"].copy()
    pivot_wer = perbin.pivot(index="file", columns="bin", values="avg_WER").reindex(columns=labels)
    pivot_cer = perbin.pivot(index="file", columns="bin", values="avg_CER").reindex(columns=labels)
    counts = perbin.pivot(index="file", columns="bin", values="n").reindex(columns=labels)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        dist_df.to_excel(writer, index=False, sheet_name="length_distribution")
        summary_df.to_excel(writer, index=False, sheet_name="summary_long")
        pivot_wer.to_excel(writer, sheet_name="avg_WER_pivot")
        pivot_cer.to_excel(writer, sheet_name="avg_CER_pivot")
        counts.to_excel(writer, sheet_name="counts_pivot")

    print(f"Saved excel: {OUT_XLSX}")


if __name__ == "__main__":
    main()
