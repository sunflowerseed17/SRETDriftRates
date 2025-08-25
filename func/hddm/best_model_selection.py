import os
import re
import numpy as np
import pandas as pd
import hddm

# Load & preprocess
df = pd.read_csv("data/SRET2019.csv")

df = df.dropna(subset=["SERT.RT", "Response"]).copy()
df["rt"] = df["SERT.RT"] / 1000.0
df["response"] = (df["Response"].astype(str).str.strip().str.lower().isin(
    ["1", "yes", "y", "true"])).astype(int)

df["subj_idx"] = df["Subject"].astype(str)
df["Valence"] = df["Valence"].astype(str).str.strip().str.lower()
df["Circumplex"] = df["Circumplex"].astype(
    str).str.strip().str.lower().str.replace("  ", " ", regex=False)
df["Circumplex_no_space"] = df["Circumplex"].str.replace(" ", "", regex=False)

keep_mask = (
    ((df["Valence"] == "positive")
     & df["Circumplex_no_space"].isin(["highaffiliation", "highdominance"])) |
    ((df["Valence"] == "negative")
     & df["Circumplex_no_space"].isin(["lowaffiliation", "lowdominance"])))
df = df.loc[keep_mask].copy()

df["Circumplex2"] = np.where(
    df["Circumplex_no_space"].str.contains("affiliation"), "affiliation",
    "dominance")

df["Valence"] = pd.Categorical(df["Valence"],
                               categories=["negative", "positive"])
df["Circumplex2"] = pd.Categorical(df["Circumplex2"],
                                   categories=["affiliation", "dominance"])

df = df[(df["rt"] > 0.15) & (df["rt"] < 5.0)].copy()

# Sanity checks
assert set(df["response"].unique()) <= {0, 1}
assert {"rt", "response", "subj_idx", "Valence",
        "Circumplex2"}.issubset(df.columns)

# Fit BEST model: v,a,t,z vary by Valence × Circumplex2
depends = {
    "v": ["Valence", "Circumplex2"],
    "a": ["Valence", "Circumplex2"],
    "t": ["Valence", "Circumplex2"],
    "z": ["Valence", "Circumplex2"],
}

print("Building HDDM (v,a,t,z ~ Valence × Circumplex2)...")
model = hddm.HDDM(
    df[["rt", "response", "subj_idx", "Valence", "Circumplex2"]],
    depends_on=depends,
    include=["v", "a", "t", "z"],
    is_group_model=True,
    p_outlier=0.05,
)

print("Finding starting values...")
model.find_starting_values()
print("Sampling...")
model.sample(4000, burn=1000, dbname=None, db=None)

print("\nModel summary:")
model.print_stats()

os.makedirs("output", exist_ok=True)
model.save("output/best_hddm_model")

# Extract per-subject parameter means per condition
traces = model.get_traces()
cols = list(traces.columns)
subjects = set(df["subj_idx"].astype(str))
VALS = ["negative", "positive"]
CIRCS = ["affiliation", "dominance"]

node_re = re.compile(r'^(?P<par>[vatz])'
                     r'(?:\((?P<f1>[^)]+)\))?'
                     r'(?:\((?P<f2>[^)]+)\))?'
                     r'_subj\.(?P<sid>.+)$')


def parse_fac(s):
    out = {}
    if not s:
        return out
    for part in re.split(r'[,\s]+', s.strip()):
        if '.' in part:
            k, v = part.split('.', 1)
            out[k.strip()] = v.strip()
    return out


rows = []
for c in cols:
    m = node_re.match(c)
    if not m:
        continue
    par = m.group("par")
    if par not in "vatz":
        continue
    sid = str(m.group("sid"))
    if sid not in subjects:
        continue

    facs = {}
    facs.update(parse_fac(m.group("f1")))
    facs.update(parse_fac(m.group("f2")))
    val = facs.get("Valence")
    cir = facs.get("Circumplex2")
    if val not in VALS or cir not in CIRCS:
        continue

    rows.append({
        "Subject": sid,
        "Valence": val,
        "Circumplex2": cir,
        "param": par,
        "mean": float(np.asarray(traces[c]).mean()),
    })

param_long = (pd.DataFrame(rows).pivot_table(
    index=["Subject", "Valence", "Circumplex2"],
    columns="param",
    values="mean",
    aggfunc="mean").reset_index().rename_axis(None, axis=1).sort_values(
        ["Subject", "Valence", "Circumplex2"]))

# Save DRIFT ONLY table


def cond_label(v, c):
    return f"{v}_{c}"


drift_rows = []
for _, r in param_long.iterrows():
    subj = r["Subject"]
    v = r["Valence"]
    c = r["Circumplex2"]
    if "v" in r and pd.notna(r["v"]):
        drift_rows.append({
            "Subject": subj,
            "Condition": cond_label(v, c),
            "Drift": float(r["v"]),
        })

drift_df = pd.DataFrame(drift_rows).sort_values(["Subject", "Condition"])

# Merge symptoms if present
symptom_cols = [
    "LSAS", "SPIN", "FPES", "BFNE", "BDI", "RSES", "STAI-S", "STAI-T"
]
have_symptoms = [c for c in symptom_cols if c in df.columns]
if have_symptoms:
    sym = df[["subj_idx"] + have_symptoms].drop_duplicates().rename(
        columns={"subj_idx": "Subject"})
    drift_df = drift_df.merge(sym, on="Subject", how="left")

drift_path = "output/dm_absolute_drift_subjects.csv"
drift_df.to_csv(drift_path, index=False)
print(f"\nSaved drift table → {drift_path}")
print(drift_df.head())

# Save full v,a,t,z per condition for each subject
full_path = "output/hddm_params_subject_long.csv"
param_long.to_csv(full_path, index=False)
print(f"Saved full subject params → {full_path}")
