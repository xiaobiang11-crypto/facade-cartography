import pandas as pd, os

p = r"F:\ucl-term1\segment-anything-main\out_all\stats_wwr.csv"
blocks_dir = r"F:\ucl-term1\segment-anything-main\out_all\blocks"

df = pd.read_csv(p).dropna(subset=["wwr"])
q20, q80 = df["wwr"].quantile([0.2, 0.8])
df["band"] = pd.cut(df["wwr"], [-1, q20, q80, 2], labels=["LOW", "MID", "HIGH"])

for band, name in [("HIGH", "wwr_high_list.csv"), ("LOW", "wwr_low_list.csv")]:
    out = df[df["band"] == band].copy()
    out["blocks_png"] = out["image"].apply(lambda s: os.path.join(blocks_dir, os.path.splitext(s)[0] + "_blocks.png"))
    out[["image", "wwr", "blocks_png"]].to_csv(os.path.join(os.path.dirname(p), name), index=False)
    print("saved:", name, "rows=", len(out))
