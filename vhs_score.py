# vhs_score.py
import os, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_in",  required=True, help="e.g. out_all/stats_wwr_shading.csv")
    ap.add_argument("--stats_out", required=True, help="e.g. out_all/stats_vhs.csv")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta",  type=float, default=0.4)
    args = ap.parse_args()

    df = pd.read_csv(args.stats_in)
    # 兼容不同列名
    wwr   = df.get("wwr", df.get("WWR"))
    shade = df.get("shading_idx", df.get("shade_idx"))

    df["wwr_01"]   = wwr.clip(0, 1)
    df["shade_01"] = shade.clip(0, 1)  # 数值越大遮阳越充分
    df["VHS"]      = args.alpha * df["wwr_01"] + args.beta * (1 - df["shade_01"])

    df.to_csv(args.stats_out, index=False)
    print("saved:", args.stats_out, "rows=", len(df))

if __name__ == "__main__":
    main()
