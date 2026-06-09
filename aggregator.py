import pandas as pd

# Load CSV
df = pd.read_csv("wandb_export.csv")

# Columns that vary by seed and should be averaged
metric_cols = [
    "train/loss",
    "val/loss",
    "train/step_loss",
    "Runtime",
]

# Configuration columns that define an experiment
group_cols = [
    "run_name",
    "activation_type",
    "batch_size",
    "block_size",
    "dropout",
    "learning_rate",
    "linear_type",
    "n_embd",
    "n_head",
    "n_layer",
    "n_params",
    "norm_placement",
    "norm_type",
    "pos_encoding",
    "residual_type",
]

# Average across seeds
summary = (
    df.groupby(group_cols)[metric_cols]
      .agg(["mean", "std"])
      .reset_index()
)

# Flatten column names
summary.columns = [
    "_".join(col).strip("_")
    if isinstance(col, tuple)
    else col
    for col in summary.columns
]

# Count how many seeds contributed
seed_counts = (
    df.groupby(group_cols)["seed"]
      .count()
      .reset_index(name="num_seeds")
)

summary = summary.merge(seed_counts, on=group_cols)

summary.to_csv("ablation_summary.csv", index=False)

print(summary.head())