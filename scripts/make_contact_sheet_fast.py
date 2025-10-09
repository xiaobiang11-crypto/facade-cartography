import pandas as pd, os 
p = r'F:\ucl-term1\segment-anything-main\out_all\stats_wwr_shading_clean.csv'
if not os.path.exists(p): p = r'F:\ucl-term1\segment-anything-main\out_all\stats_wwr_shading_clean.csv'
df = pd.read_csv(p)
q80 = df['wwr'].quantile(0.8); q20 = df['shading_idx'].quantile(0.2)
risk = df[(df['wwr']>=q80) & (df['shading_idx']<=q20)].copy()
risk.sort_values(['wwr','shading_idx'], ascending=[False,True], inplace=True)
out_csv = os.path.join(os.path.dirname(p), 'priority_wwrHigh_shadeLow_clean.csv')
risk.to_csv(out_csv, index=False); print('saved:', out_csv, 'rows=', len(risk))
PY

python .\make_contact_sheet.py --images_dir F:\ucl-term1\segment-anything-main\out_all\blocks ^
  --csv F:\ucl-term1\segment-anything-main\out_all\priority_wwrHigh_shadeLow_clean.csv ^
  --sort_by wwr --count 60 --cols 10 ^
  --out F:\ucl-term1\segment-anything-main\out_all\contact_priority_wwrHigh_shadeLow_clean.png
@'
import os, math, argparse, pandas as pd
from PIL import Image, ImageDraw

def find_image(row, root):
    # 尽量兼容不同CSV列名：blocks_png / image
    cand = []
    if "blocks_png" in row and isinstance(row["blocks_png"], str):
        cand.append(row["blocks_png"])
    if "image" in row and isinstance(row["image"], str):
        stem = os.path.splitext(os.path.basename(row["image"]))[0]
        cand += [f"{stem}.png", f"{stem}_blocks.png"]
    for c in cand:
        p = c if os.path.isabs(c) else os.path.join(root, c)
        if os.path.exists(p): return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sort_by", default=None)
    ap.add_argument("--ascending", action="store_true")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--thumb", type=int, default=180)   # 缩略图边长，调小更快
    ap.add_argument("--pad", type=int, default=6)       # 留白
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending)
    df = df.head(args.count).reset_index(drop=True)

    # 读取与缩放
    thumbs = []
    for _, r in df.iterrows():
        p = find_image(r, args.images_dir)
        if not p: continue
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((args.thumb, args.thumb))  # 正方形缩略图
            thumbs.append(im)
        except Exception:
            continue

    if not thumbs:
        raise SystemExit("No images found to compose contact sheet.")

    cols = args.cols
    rows = math.ceil(len(thumbs) / cols)
    W = cols * args.thumb + (cols + 1) * args.pad
    H = rows * args.thumb + (rows + 1) * args.pad

    canvas = Image.new("RGB", (W, H), (8, 24, 120))  # 深蓝底，和你的配色一致
    draw = ImageDraw.Draw(canvas)

    for i, im in enumerate(thumbs):
        r = i // cols; c = i % cols
        x0 = args.pad + c * (args.thumb + args.pad)
        y0 = args.pad + r * (args.thumb + args.pad)
        canvas.paste(im, (x0, y0))
        # 细白框
        draw.rectangle([x0-1, y0-1, x0+args.thumb, y0+args.thumb], outline=(240,240,255))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out, format="PNG", optimize=True)
    print("saved:", args.out, "tiles=", len(thumbs), "grid=", rows, "x", cols)

if __name__ == "__main__":
    main()
