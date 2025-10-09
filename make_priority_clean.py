import pandas as pd, os
p = r"F:\ucl-term1\segment-anything-main\out_all\stats_wwr_shading_clean.csv"
df = pd.read_csv(p)
q80 = df["wwr"].quantile(0.8); q20 = df["shading_idx"].quantile(0.2)
risk = df[(df["wwr"]>=q80) & (df["shading_idx"]<=q20)].copy()
risk.sort_values(["wwr","shading_idx"], ascending=[False,True], inplace=True)
out_csv = os.path.join(os.path.dirname(p), "priority_wwrHigh_shadeLow_clean.csv")
risk.to_csv(out_csv, index=False)
print("saved:", out_csv, "rows=", len(risk))
