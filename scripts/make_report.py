import os, pandas as pd

root = r"F:\ucl-term1\segment-anything-main\out_all"
# 读入
stats      = pd.read_csv(os.path.join(root,"stats.csv"))                  # 行列/规则性等
metrics    = pd.read_csv(os.path.join(root,"stats_metrics.csv"))          # 占比/面积等
wwr        = pd.read_csv(os.path.join(root,"stats_wwr.csv"))              # WWR
ws_clean   = pd.read_csv(os.path.join(root,"stats_wwr_shading_clean.csv"))# WWR+遮阳(裁剪后)
bad        = pd.read_csv(os.path.join(root,"bad_metrics.csv")) if os.path.exists(os.path.join(root,"bad_metrics.csv")) else pd.DataFrame(columns=["image"])

# 合并（按 image 对齐）
df = stats.merge(metrics, on="image", how="left")\
          .merge(wwr, on="image", how="left")\
          .merge(ws_clean, on="image", how="left", suffixes=("", "_ws"))

# 风险阈值（来自 clean 表）
q80 = df["wwr"].quantile(0.8)
q20 = df["shading_idx"].quantile(0.2)

df["risk_flag"] = (df["wwr"]>=q80) & (df["shading_idx"]<=q20)
df["bad_flag"]  = df["image"].isin(bad["image"]) if len(bad) else False

out = os.path.join(root,"report_all.csv")
df.to_csv(out, index=False)
print("saved:", out)
print("images:", len(df), "high-risk:", int(df['risk_flag'].sum()), "bad:", int(df['bad_flag'].sum()))
