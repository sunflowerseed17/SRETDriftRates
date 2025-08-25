import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as tbl
from scipy import stats

# File paths
INPUT_CSV = "output/hddm/hddm_result_analysis/dm_absolute_drift_subjects.csv"

# put all outputs in the SAME folder as the CSV
OUT_DIR = os.path.dirname(os.path.abspath(INPUT_CSV)) or "."
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

# expected columns: Subject, Condition, Drift, (BDI, LSAS optionally)
for col in ["Subject", "Condition", "Drift"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {INPUT_CSV}")

df["Subject"] = df["Subject"].astype(str)
df["Condition"] = df["Condition"].astype(str)

# canonical order
COND_ORDER = [
    "negative_affiliation",
    "negative_dominance",
    "positive_affiliation",
    "positive_dominance",
]
# keep seen conditions, but respect our preferred order
conditions = [c for c in COND_ORDER if c in df["Condition"].unique()
              ] + [c for c in df["Condition"].unique() if c not in COND_ORDER]

# Add Valence label for convenience
df["Valence"] = np.where(df["Condition"].str.startswith("negative"),
                         "negative", "positive")

# Colors (pastel, consistent)
#    - Negative: blues
#    - Positive: greens

colors = {
    "negative_affiliation": "#a4d1f6",
    "negative_dominance": "#7e99fdaa",
    "positive_affiliation": "#d9f5ca",  #
    "positive_dominance": "#66b86ac5",  #
    # valence-only
    "negative": "#234e59ce",
    "positive": "#34862e55",
}

# Shared histogram settings (same x-axis for easy comparison)
# compute a global range with a small margin

drift_all = df["Drift"].astype(float).dropna()
dr_min, dr_max = drift_all.min(), drift_all.max()
margin = 0.05 * (dr_max - dr_min) if np.isfinite(dr_max - dr_min) else 1.0
X_MIN, X_MAX = dr_min - margin, dr_max + margin

# choose fixed bins across all plots
N_BINS = 15
bins = np.linspace(X_MIN, X_MAX, N_BINS + 1)

title_font = {"fontsize": 18, "fontweight": "bold"}
label_font = {"fontsize": 13}
tick_fontsize = 12


def tidy_title(s: str) -> str:
    """Turn 'negative_affiliation' into 'Negative • Affiliation' for titles."""
    if "_" in s:
        a, b = s.split("_", 1)
        return f"{a.capitalize()} • {b.capitalize()}"
    return s.capitalize()


# 4) Histograms by Condition (with shared x-lims)
for cond in conditions:
    sub = df.loc[df["Condition"] == cond, "Drift"].astype(float).dropna()
    if sub.empty:
        continue

    plt.figure(figsize=(10, 6))
    plt.hist(sub,
             bins=bins,
             color=colors.get(cond, "#888888"),
             edgecolor="black",
             alpha=0.9)
    plt.title(f"Drift rate distribution — {tidy_title(cond)}", **title_font)
    plt.xlabel("Drift rate (v)", **label_font)
    plt.ylabel("Count", **label_font)
    plt.xlim(X_MIN, X_MAX)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_{cond}.png"), dpi=300)
    plt.close()

# Histograms by Valence (pooled across affiliation/dominance)
for val in ["negative", "positive"]:
    sub = df.loc[df["Valence"] == val, "Drift"].astype(float).dropna()
    if sub.empty:
        continue

    plt.figure(figsize=(10, 6))
    plt.hist(sub, bins=bins, color=colors[val], edgecolor="black", alpha=0.9)
    plt.title(f"Drift rate distribution — {val.capitalize()}", **title_font)
    plt.xlabel("Drift rate (v)", **label_font)
    plt.ylabel("Count", **label_font)
    plt.xlim(X_MIN, X_MAX)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_valence_{val}.png"), dpi=300)
    plt.close()

print("Saved histograms with a shared x-axis scale.")


# Regressions helper
def regress_one(x: pd.Series, y: pd.Series):
    """Return r, p, R2, f2 (NaN-safe)."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    x, y = x.dropna(), y.dropna()
    # align by index intersection
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]
    if x.nunique() < 2 or len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    R2 = r**2
    f2 = R2 / (1 - R2) if 0 <= R2 < 1 else np.nan
    return r, p, R2, f2


def regression_table(groupby_col: str, symptom_cols=("BDI", "LSAS")):
    """Build a regression table grouped by 'Condition' or 'Valence'."""
    rows = []
    for level, g in df.groupby(groupby_col):
        row = {groupby_col: level}
        for sym in symptom_cols:
            if sym in g.columns:
                r, p, R2, f2 = regress_one(g[sym], g["Drift"])
                row.update({
                    f"{sym}_r": r,
                    f"{sym}_p": p,
                    f"{sym}_R2": R2,
                    f"{sym}_f2": f2
                })
            else:
                row.update({
                    f"{sym}_r": np.nan,
                    f"{sym}_p": np.nan,
                    f"{sym}_R2": np.nan,
                    f"{sym}_f2": np.nan
                })
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(groupby_col)
    return out


# condition-level and valence-level tables
tbl_cond = regression_table("Condition")
tbl_val = regression_table("Valence")

# persist raw numeric tables (optional, for record)
tbl_cond.to_csv(os.path.join(OUT_DIR, "drift_regressions_by_condition.csv"),
                index=False)
tbl_val.to_csv(os.path.join(OUT_DIR, "drift_regressions_by_valence.csv"),
               index=False)

# Nicely formatted table figures with p-value shading
#     (short headers so text always fits)


def save_reg_table_png(df_in: pd.DataFrame, index_col: str, out_png: str,
                       title: str):
    df = df_in.copy()
    # pretty rounding
    for c in df.columns:
        if c == index_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].round(3)

    columns = [
        index_col, "r BDI", "p BDI", "R² BDI", "f² BDI", "r LSAS", "p LSAS",
        "R² LSAS", "f² LSAS"
    ]
    rename_map = {
        "BDI_r": "r BDI",
        "BDI_p": "p BDI",
        "BDI_R2": "R² BDI",
        "BDI_f2": "f² BDI",
        "LSAS_r": "r LSAS",
        "LSAS_p": "p LSAS",
        "LSAS_R2": "R² LSAS",
        "LSAS_f2": "f² LSAS"
    }
    df = df.rename(columns=rename_map)
    df = df[[c for c in columns if c in df.columns]]

    data = df.values.tolist()
    ncols = len(df.columns)
    fig_w = 2.1 * ncols + 2.0
    fig_h = 1.0 + 0.45 * max(len(df), 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.30] + [0.14] * (ncols - 1)

    table = tbl.table(ax,
                      cellText=[df.columns.tolist()] + data,
                      cellLoc="center",
                      colWidths=col_widths,
                      loc="center")
    # header style
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", fontsize=12)
            cell.set_height(0.5)
        else:
            cell.set_fontsize(11)
            # p-value shading
            try:
                header = df.columns[c]
                val = float(cell.get_text().get_text())
                if header.startswith("p "):
                    if val <= 0.001:
                        cell.set_facecolor("#D110107E")  # strong
                    elif val <= 0.01:
                        cell.set_facecolor("#FF8C8C90")  # moderate
                    elif val <= 0.05:
                        cell.set_facecolor("#EEB9B486")  # weak
                    else:
                        cell.set_facecolor("#FFFFFF")  # ns
            except Exception:
                pass

    # title
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)

    # small legend
    legend_lines = [
        ("p ≤ .001", "#D110107E"),
        ("p ≤ .01", "#FF8C8C90"),
        ("p ≤ .05", "#EEB9B486"),
        ("ns", "#FFFFFF"),
    ]
    x0, y0 = 0.01, 0.02
    for i, (lab, col) in enumerate(legend_lines):
        y = y0 + i * 0.04
        ax.add_patch(
            plt.Rectangle((x0, y),
                          0.02,
                          0.03,
                          transform=ax.transAxes,
                          facecolor=col,
                          edgecolor="black",
                          lw=0.5))
        ax.text(x0 + 0.025,
                y + 0.015,
                lab,
                transform=ax.transAxes,
                va="center",
                fontsize=10)

    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(OUT_DIR, out_png), dpi=300)
    plt.close()


save_reg_table_png(tbl_cond, "Condition",
                   "drift_regression_table_by_condition.png",
                   "Drift–symptom regressions by Condition")

save_reg_table_png(tbl_val, "Valence", "drift_regression_table_by_valence.png",
                   "Drift–symptom regressions by Valence")

print(f"All outputs written to: {OUT_DIR}")
