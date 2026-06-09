import wandb
import pandas as pd

api = wandb.Api()

PROJECT = "madhavkrishnan747-australian-national-university/nanogpt-ablations"

runs = api.runs(PROJECT)

rows = []

for run in runs:
    print(f"Downloading {run.name}")

    try:
        for record in run.scan_history():
            record["run_id"] = run.id
            record["run_name"] = run.name

            for k, v in run.config.items():
                if not k.startswith("_"):
                    record[f"config.{k}"] = v

            rows.append(record)

    except Exception as e:
        print(f"Error: {e}")

df = pd.DataFrame(rows)
df.to_csv("wandb_full_history.csv", index=False)

print(f"Saved {len(df):,} rows")