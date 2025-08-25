import os
import re
import numpy as np
import pandas as pd
import hddm

# Load and preprocess
df = pd.read_csv("data/SRET2019.csv")

# Basic cleaning
df = df.dropna(subset=["SERT.RT", "Response"]).copy()
df["rt"] = df["SERT.RT"] / 1000.0

# Choice -> 0/1 (int)
df["choice"] = (df["Response"].astype(str).str.strip().str.lower().isin(
    ["1", "yes", "y", "true"])).astype(int)
df["response"] = df["choice"].astype(int)

# IDs and factors
df["subj_idx"] = df["Subject"].astype(str)
df["Valence"] = df["Valence"].astype(str).str.strip().str.lower()
df["Circumplex"] = (
    df["Circumplex"].astype(str).str.strip().str.lower().str.replace(
        "  ", " ", regex=False))
df["Circumplex_no_space"] = df["Circumplex"].str.replace(" ", "", regex=False)

keep_mask = (
    ((df["Valence"] == "positive")
     & df["Circumplex_no_space"].isin(["highaffiliation", "highdominance"])) |
    ((df["Valence"] == "negative")
     & df["Circumplex_no_space"].isin(["lowaffiliation", "lowdominance"])))
df = df.loc[keep_mask].copy()

# Collapse to two-level Circumplex: affiliation vs dominance
df["Circumplex2"] = np.where(
    df["Circumplex_no_space"].str.contains("affiliation"), "affiliation",
    "dominance")

# Order factors to match brms levels
df["Valence"] = pd.Categorical(df["Valence"],
                               categories=["negative", "positive"])
df["Circumplex2"] = pd.Categorical(df["Circumplex2"],
                                   categories=["affiliation", "dominance"])

# Sum-to-zero contrasts (±1)
df["V"] = df["Valence"].map({"negative": -1, "positive": 1}).astype(int)
df["C"] = df["Circumplex2"].map({
    "affiliation": -1,
    "dominance": 1
}).astype(int)

# Optional: drop extreme/invalid RTs
df = df[(df["rt"] > 0.15) & (df["rt"] < 5.0)].copy()

# Checks

assert set(df["response"].unique()).issubset({0,
                                              1}), "response must be 0/1 ints"
assert df["V"].nunique() == 2 and df["C"].nunique(
) == 2, "V and C must vary and be ±1"
assert {"rt", "response", "subj_idx", "V", "C"}.issubset(df.columns)

# Minimal data for HDDM
data = df[["rt", "response", "subj_idx", "V", "C"]].copy()
data["response"] = data["response"].astype(int)

# HDDMRegressor spec
# This gives subject-specific intercepts and slopes (random effects),
# which HDDM exposes as v_Intercept_subj.*, v_V_subj.*, v_C_subj.*, v_V:C_subj.*
reg_spec = [{"model": "v ~ 1 + V + C + V:C", "link_func": lambda x: x}]

model = hddm.HDDMRegressor(data,
                           reg_spec,
                           group_only_regressors=False,
                           include=["v", "a", "t", "z"],
                           p_outlier=0.05)

print("Finding starting values...")
model.find_starting_values()
print("Sampling...")
model.sample(4000, burn=1000, dbname="hddm_traces.db", db="pickle")

print("\nModel summary:")
model.print_stats()

# Trace helpers
_traces_df = model.get_traces()


def trace_mean_exact(name: str):
    try:
        return float(np.asarray(_traces_df[name]).mean())
    except Exception:
        return None


def cols_like(pattern: str):
    # simple glob: pattern may contain '*' to match any chars
    if "*" in pattern:
        regex = "^" + pattern.replace(".", r"\.").replace("*", ".*") + "$"
        return [c for c in _traces_df.columns if re.match(regex, c)]
    return [c for c in _traces_df.columns if c == pattern]


def trace_cols():
    return list(_traces_df.columns)


# Group-level fixed effects
b0 = trace_mean_exact("v_Intercept")
bV = trace_mean_exact("v_V")
bC = trace_mean_exact("v_C")
bVC = trace_mean_exact("v_V:C")

if None in (b0, bV, bC, bVC):
    print(
        "\n[WARN] Some fixed-effect names were not found in traces. First 20 columns:"
    )
    print(", ".join(trace_cols()[:20]))
else:
    print(
        f"\nGroup fixed effects (sum-coded): b0={b0:.3f}, bV={bV:.3f}, bC={bC:.3f}, bVxC={bVC:.3f}"
    )

# Collect SUBJECT-SPECIFIC coefficients
# We want the actual subject coefficients, not just deviations.
# HDDM prints them as v_Intercept_subj.XXXX, v_V_subj.XXXX, etc.

all_cols = trace_cols()
id_set = set(df["subj_idx"].astype(str))


def extract_subj_coef(prefix: str):
    """Return dict: subj_id -> posterior mean for columns like 'prefix_subj.<ID>'"""
    out = {}
    # try the common pattern first
    patt = re.compile(rf"^{re.escape(prefix)}_subj\.(.+)$")
    for c in all_cols:
        m = patt.match(c)
        if m:
            suffix = str(m.group(1))
            val = trace_mean_exact(c)
            if val is None:
                continue
            # If suffix matches a real subject label, use it directly.
            if suffix in id_set:
                subj_id = suffix
            else:
                # Fallback: treat suffix as 1-based positional index into sorted unique subjects.
                levels = sorted(df["subj_idx"].unique().tolist())
                try:
                    idx = int(suffix) - 1
                    subj_id = levels[idx] if 0 <= idx < len(levels) else suffix
                except ValueError:
                    subj_id = suffix
            out[str(subj_id)] = float(val)
    return out


b0_subj = extract_subj_coef("v_Intercept")
bV_subj = extract_subj_coef("v_V")
bC_subj = extract_subj_coef("v_C")
bVC_subj = extract_subj_coef("v_V:C")

# Compute per-subject drift per condition USING SUBJECT COEFS

conditions = {
    "negative_affiliation": (-1, -1),
    "negative_dominance": (-1, +1),
    "positive_affiliation": (+1, -1),
    "positive_dominance": (+1, +1),
}

rows = []
subjects = sorted(df["subj_idx"].astype(str).unique().tolist())

missing_any = []
for subj in subjects:
    if not ({subj}.issubset(b0_subj.keys()) and {subj}.issubset(bV_subj.keys())
            and {subj}.issubset(bC_subj.keys())
            and {subj}.issubset(bVC_subj.keys())):
        missing_any.append(subj)
        continue

    b0i = b0_subj[subj]
    bVi = bV_subj[subj]
    bCi = bC_subj[subj]
    bVCi = bVC_subj[subj]

    for name, (v_code, c_code) in conditions.items():
        drift = b0i + bVi * v_code + bCi * c_code + bVCi * (v_code * c_code)
        rows.append({
            "Subject": subj,
            "Condition": name,
            "Drift": float(drift)
        })

if missing_any:
    print(
        f"\n[WARN] Missing subject coefficients for {len(missing_any)} subjects; "
        f"they will be omitted from the output. Example: {missing_any[:5]}")

subject_cond = pd.DataFrame(rows).sort_values(["Subject", "Condition"])

# Merge symptoms

symptom_cols = [
    "LSAS", "SPIN", "FPES", "BFNE", "BDI", "RSES", "STAI-S", "STAI-T"
]
symptoms = (
    df[["subj_idx"] +
       [c for c in symptom_cols if c in df.columns]].drop_duplicates().rename(
           columns={"subj_idx": "Subject"}))

out = subject_cond.merge(symptoms, on="Subject", how="left")
os.makedirs("output", exist_ok=True)
out.to_csv("output/dm_absolute_drift_subjects.csv", index=False)
print("\nSaved → output/dm_absolute_drift_subjects.csv")
print(out.head())
