import argparse, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csv_in", required=True)
ap.add_argument("--csv_out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.csv_in)

# 计算 occupancy = blocks / (rows * median_cols_per_row)
grid = df["rows"].astype(float) * df["median_cols_per_row"].astype(float)
df["occupancy"] = df["blocks_drawn"].astype(float) / grid.replace(0, np.nan)

# 一个简易的规整度指数（0~1，越大越规整）——先做个可用版本
r = df["rows"]; c = df["median_cols_per_row"]; s = (df["occupancy"]-1).abs()
def norm(x): return (x - x.median()).abs() / (x.quantile(0.75)-x.quantile(0.25)+1e-6)
r_norm = 1 - norm(r); c_norm = 1 - norm(c)
s_norm = 1 - (s - s.min()) / (s.max()-s.min()+1e-6)
df["regularity_index"] = (0.4*r_norm + 0.4*c_norm + 0.2*s_norm).clip(0,1)

df.to_csv(args.csv_out, index=False)
print("saved:", args.csv_out)
