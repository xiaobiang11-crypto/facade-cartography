import pandas as pd, matplotlib.pyplot as plt, os

p = r"F:\ucl-term1\segment-anything-main\out_all\stats_wwr_shading.csv"
df = pd.read_csv(p).copy()
df["wwr"] = df["wwr"].clip(0, 1)
df["shading_idx"] = df["shading_idx"].clip(0, 1)

pc = p.replace(".csv", "_clean.csv")
df.to_csv(pc, index=False)
print("saved:", pc)

q80 = df["wwr"].quantile(0.8)
q20 = df["shading_idx"].quantile(0.2)

plt.figure(figsize=(6,5))
plt.scatter(df["wwr"], df["shading_idx"], s=12, alpha=0.6)
plt.axvline(q80, ls="--", lw=1)
plt.axhline(q20, ls="--", lw=1)
plt.xlim(0,1); plt.ylim(0,1)
plt.xlabel("WWR"); plt.ylabel("Shading index")
plt.title("WWR vs Shading (risk = high WWR & low shading)")

out = p.replace(".csv", "_wwr_vs_shading_clean.png")
plt.tight_layout()
plt.savefig(out, dpi=160)
print("saved:", out)
