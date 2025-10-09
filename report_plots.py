import os, pandas as pd, matplotlib.pyplot as plt
CSV = r"F:\ucl-term1\segment-anything-main\out_all\stats.csv"
df = pd.read_csv(CSV)

# rows / median_cols_per_row 的直方图
for col, title in [("rows","Rows per facade"),("median_cols_per_row","Median columns per row")]:
    plt.figure()
    df[col].hist(bins=20)
    plt.title(title); plt.xlabel(col); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(CSV), f"{col}_hist.png"), dpi=200)

# blocks_drawn vs rows*cols 的散点（粗看“密度”）
if {"blocks_drawn","rows","median_cols_per_row"} <= set(df.columns):
    plt.figure()
    plt.scatter(df["rows"]*df["median_cols_per_row"], df["blocks_drawn"], s=12)
    plt.xlabel("rows * median_cols"); plt.ylabel("blocks_drawn")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(CSV), "blocks_vs_grid.png"), dpi=200)
print("Saved plots to:", os.path.dirname(CSV))
