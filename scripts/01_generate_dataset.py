# Import needed libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Add random seed to ensure reproducibility
np.random.seed(1990)
N = 300  # sample size

####  Context definition (Dadaab & Kakuma)
refugee_camp = ["Dadaab", "Kakuma"]
sex = ["Male", "Female"]
age_group = ["6–9", "10–13", "14–17"]

sex = ["Male", "Female"]
age_group = ["6–9", "10–13", "14–17"]

# Camp complexes
camp_complexes = ["Dadaab Camp", "Kakuma Camp"]

# sections
dadaab_sites = ["Hagadera", "Dagahaley", "Ifo"]
kakuma_sites = [
    "Kakuma 1",
    "Kakuma 2",
    "Kakuma 3",
    "Kakuma 4",
    "Kalobeyei Settlement"
]

# Split learners across complexes
complex_probs = [0.65, 0.35]  # Dadaab, Kakuma

# Within-complex allocation
dadaab_probs = [0.36, 0.34, 0.30]  # Hagadera, Dagahaley, Ifo
kakuma_probs = [0.18, 0.19, 0.18, 0.15, 0.30]  # K1, K2, K3, K4, Kalobeyei

# Create base roster
df = pd.DataFrame({
    "learner_id": range(1, N + 1),
    "complex": np.random.choice(camp_complexes, size=N, p=complex_probs),
    "sex": np.random.choice(sex, size=N, p=[0.50, 0.50]),
    "age_group": np.random.choice(age_group, size=N, p=[0.35, 0.40, 0.25])
})

# Section based on complex
df["site"] = None

dadaab_mask = df["complex"].eq("Dadaab Complex")
kakuma_mask = df["complex"].eq("Kakuma Complex")

df.loc[dadaab_mask, "site"] = np.random.choice(
    dadaab_sites, size=dadaab_mask.sum(), p=dadaab_probs
)

df.loc[kakuma_mask, "site"] = np.random.choice(
    kakuma_sites, size=kakuma_mask.sum(), p=kakuma_probs
)

# Baseline & endline indicators + field noise

# Baseline attendance (% days attended in last 20 learning days)
# Differences by complex/site are represented lightly through random variation.
base_att = np.clip(np.random.normal(loc=55, scale=15, size=N), 5, 95)

# Improvement at endline (programme effect), capped realistically
att_gain = np.clip(np.random.normal(loc=12, scale=10, size=N), -10, 35)
end_att = np.clip(base_att + att_gain, 5, 98)

# Baseline learning score (0–100), modestly associated with attendance
base_score = np.clip(np.random.normal(loc=45, scale=12, size=N) + (base_att - 55) * 0.15, 5, 95)

# Endline learning score improvement
score_gain = np.clip(np.random.normal(loc=10, scale=9, size=N), -8, 28)
end_score = np.clip(base_score + score_gain, 5, 98)

df["attendance_baseline_pct"] = base_att.round(1)
df["attendance_endline_pct"] = end_att.round(1)
df["score_baseline"] = base_score.round(0).astype(int)
df["score_endline"] = end_score.round(0).astype(int)

# Dropout flag (about 10% overall, slightly higher among older learners)
dropout_prob = np.where(df["age_group"].eq("14–17"), 0.14, 0.09)
df["dropped_out"] = (np.random.rand(N) < dropout_prob).astype(int)

dropout_reasons = [
    "Household responsibilities",
    "Moved/relocated",
    "Livelihood work",
    "Safety/protection concern",
    "Illness",
    "Other/unknown"
]

# Assign dropout reasons only where dropped_out == 1
df["dropout_reason"] = np.where(
    df["dropped_out"].eq(1),
    np.random.choice(dropout_reasons, size=N, p=[0.22, 0.20, 0.18, 0.15, 0.15, 0.10]),
    ""
)

# If dropped out, endline measures may be missing (common in follow-up loss)
df.loc[df["dropped_out"].eq(1), ["attendance_endline_pct", "score_endline"]] = np.nan

# Introduce additional missingness (~5%) to mimic incomplete registers
for col in ["attendance_baseline_pct", "score_baseline"]:
    miss_mask = np.random.rand(N) < 0.05
    df.loc[miss_mask, col] = np.nan


# Export datasets (row-level + Tableau-ready aggregates)

# Ensure data folder exists
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# 1) Save the row-level dataset
row_level_path = data_dir / "learners_kakuma_dadaab_sample.csv"
df.to_csv(row_level_path, index=False)

# 2) Create Tableau-ready aggregates
# Attendance and score improvements (endline - baseline), ignoring missing values
df["attendance_change"] = df["attendance_endline_pct"] - df["attendance_baseline_pct"]
df["score_change"] = df["score_endline"] - df["score_baseline"]

agg = (
    df.groupby(["complex", "site", "sex", "age_group"], dropna=False)
      .agg(
          learners=("learner_id", "count"),
          dropout_rate=("dropped_out", "mean"),
          baseline_attendance_mean=("attendance_baseline_pct", "mean"),
          endline_attendance_mean=("attendance_endline_pct", "mean"),
          attendance_change_mean=("attendance_change", "mean"),
          baseline_score_mean=("score_baseline", "mean"),
          endline_score_mean=("score_endline", "mean"),
          score_change_mean=("score_change", "mean")
      )
      .reset_index()
)

# Make rates and means presentation-ready
agg["dropout_rate"] = (agg["dropout_rate"] * 100).round(1)
for c in [
    "baseline_attendance_mean", "endline_attendance_mean", "attendance_change_mean",
    "baseline_score_mean", "endline_score_mean", "score_change_mean"
]:
    agg[c] = agg[c].round(1)

agg_path = data_dir / "tableau_agg_kakuma_dadaab_sample.csv"
agg.to_csv(agg_path, index=False)

