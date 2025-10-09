import os, math, argparse
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

def load_order(images_dir, csv, count, sort_by, ascending):
    if csv:
        df = pd.read_csv(csv)
        if sort_by == "occupancy" and "occupancy" not in df.columns:
            if {"rows","median_cols_per_row","blocks_drawn"} <= set(df.columns):
                grid = df["rows"].astype(float) * df["median_cols_per_row"].astype(float)
                df["occupancy"] = df["blocks_drawn"].astype(float) / grid.replace(0, np.nan)
        df = df.sort_values(sort_by, ascending=ascending)

        paths, rows, cols, blocks, occs, stems = [], [], [], [], [], []
        for _, r in df.iterrows():
            stem = os.path.splitext(str(r["image"]))[0]
            p = os.path.join(images_dir, f"{stem}_blocks.png")
            if os.path.exists(p):
                paths.append(p)
                stems.append(stem)
                rows.append(r.get("rows", np.nan))
                cols.append(r.get("median_cols_per_row", np.nan))
                blocks.append(r.get("blocks_drawn", np.nan))
                occs.append(r.get("occupancy", np.nan))
            if len(paths) >= count: break
        meta = list(zip(stems, rows, cols, blocks, occs))
        return paths, meta
    else:
        names = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(".png")]
        paths = [os.path.join(images_dir, f) for f in names[:count]]
        meta = [(os.path.splitext(os.path.basename(p))[0], np.nan, np.nan, np.nan, np.nan) for p in paths]
        return paths, meta

def make_sheet(paths, meta, cols, thumb_w, thumb_h, out_path,
               pad=16, bg=(10,30,210), border=2, label_bar=24, font_path=None):
    if not paths: 
        print("No images to place."); return
    rows = math.ceil(len(paths)/cols)
    W = cols*thumb_w + (cols+1)*pad
    H = rows*(thumb_h+label_bar) + (rows+1)*pad
    sheet = Image.new("RGB", (W,H), bg)
    try:
        font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    for i,(p,info) in enumerate(zip(paths, meta)):
        im = Image.open(p).convert("RGB").resize((thumb_w,thumb_h), Image.BILINEAR)
        r, c = divmod(i, cols)
        x = pad + c*(thumb_w+pad); y = pad + r*(thumb_h+label_bar+pad)

        # 贴图
        sheet.paste(im, (x,y))

        # 白色细边框
        draw = ImageDraw.Draw(sheet)
        draw.rectangle([x, y, x+thumb_w-1, y+thumb_h-1], outline=(240,240,240), width=border)

        # 标签条
        lx0, ly0 = x, y+thumb_h
        draw.rectangle([lx0, ly0, lx0+thumb_w, ly0+label_bar], fill=(15,15,25))

        stem, rws, cls, blks, occ = info
        if np.isnan(occ): lbl = stem
        else:             lbl = f"{stem} | r{int(rws)}×c{int(cls)} | b{int(blks)} | occ {occ:.2f}"
        draw.text((lx0+6, ly0+4), lbl, fill=(255,255,255), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print("Saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)    # e.g. ...\out_all\blocks
    ap.add_argument("--csv", default=None)            # e.g. ...\out_all\stats.csv
    ap.add_argument("--sort_by", default="occupancy",
                    choices=["occupancy","blocks_drawn","rows","median_cols_per_row"])
    ap.add_argument("--ascending", action="store_true")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--thumb_w", type=int, default=240)
    ap.add_argument("--thumb_h", type=int, default=320)
    ap.add_argument("--out", default=None)
    ap.add_argument("--font", default=None)           # 可传入 .ttf 路径，美观些
    args = ap.parse_args()

    paths, meta = load_order(args.images_dir, args.csv, args.count, args.sort_by, args.ascending)
    if not args.out:
        base = f"contact_by_{args.sort_by}_{'asc' if args.ascending else 'desc'}.png" if args.csv else "contact_sheet.png"
        args.out = os.path.join(os.path.dirname(args.images_dir), base)

    make_sheet(paths, meta, args.cols, args.thumb_w, args.thumb_h, args.out, font_path=args.font)

if __name__ == "__main__":
    main()
