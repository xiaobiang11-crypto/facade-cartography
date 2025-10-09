import os, math, argparse
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, numpy as np

def load_order(images_dir, csv, count, sort_by, ascending):
    if csv:
        df = pd.read_csv(csv)
        if sort_by == "occupancy" and "occupancy" not in df.columns:
            if {"rows","median_cols_per_row","blocks_drawn"} <= set(df.columns):
                grid = df["rows"].astype(float)*df["median_cols_per_row"].astype(float)
                df["occupancy"] = df["blocks_drawn"].astype(float)/grid.replace(0,np.nan)
        df = df.sort_values(sort_by, ascending=ascending)
        paths, meta = [], []
        for _, r in df.iterrows():
            stem = os.path.splitext(str(r["image"]))[0]
            p = os.path.join(images_dir, f"{stem}_blocks.png")
            if os.path.exists(p):
                paths.append(p)
                meta.append(stem)
            if len(paths) >= count: break
        return paths, meta
    else:
        names = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(".png")]
        paths = [os.path.join(images_dir, f) for f in names[:count]]
        return paths, [os.path.splitext(os.path.basename(p))[0] for p in paths]

def make_sheet(paths, labels, cols, tw, th, out_path, pad=16, bg=(10,30,210)):
    rows = math.ceil(len(paths)/cols)
    W = cols*tw + (cols+1)*pad
    H = rows*th + (rows+1)*pad
    sheet = Image.new("RGB", (W,H), bg)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i,(p,lab) in enumerate(zip(paths,labels)):
        im = Image.open(p).convert("RGB").resize((tw,th), Image.BILINEAR)
        r,c = divmod(i,cols)
        x = pad + c*(tw+pad); y = pad + r*(th+pad)
        sheet.paste(im,(x,y))
        draw.rectangle([x,y,x+tw-1,y+th-1], outline=(240,240,240), width=2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path); print("Saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sort_by", default="occupancy")
    ap.add_argument("--ascending", action="store_true")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--thumb_w", type=int, default=240)
    ap.add_argument("--thumb_h", type=int, default=320)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths, labels = load_order(args.images_dir, args.csv, args.count, args.sort_by, args.ascending)
    if not args.out:
        base = f"contact_by_{args.sort_by}_{'asc' if args.ascending else 'desc'}.png" if args.csv else "contact_sheet.png"
        args.out = os.path.join(os.path.dirname(args.images_dir), base)
    make_sheet(paths, labels, args.cols, args.thumb_w, args.thumb_h, args.out)

if __name__ == "__main__":
    main()
