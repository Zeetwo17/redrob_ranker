"""Convert submission CSV to XLSX format."""
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="outputs/submission.csv")
parser.add_argument("--out", default="outputs/submission.xlsx")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df.to_excel(args.out, index=False)
print(f"Wrote {len(df)} rows to {args.out}")
