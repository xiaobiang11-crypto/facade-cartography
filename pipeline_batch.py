import os, argparse, glob, math, csv
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
from skimage.morphology import binary_closing, square
from sklearn.cluster import AgglomerativeClustering

# ---------------- 参数默认值（可在命令行覆盖） ----------------
PALETTE = [(26,54,255),(17,132,255),(0,191,255),(0,230,171),
           (148,231,0),(227,211,0),(255,153,0),(255,66,0)]
ROW_MERGE_PX = 22       # 行聚类阈值（像素）
COL_MERGE_PX = 22       # 列聚类阈值（像素）
MORPH_CLOSE  = 3        # 掩膜闭运算窗口（合并碎片），设 0/1 关闭
# “窗口/门/阳台”候选过滤（按图可调）
AREA_MIN_PCT = 0.02     # mask 面积占整图百分比下限
AREA_MAX_PCT = 9.0      # 上限
FILL_MIN     = 0.28     # 填充比 area/bbox_area 下限
FILL_MAX     = 0.95     # 上限
ASPECT_MIN   = 0.28     # 宽高比 w/h 下限
ASPECT_MAX   = 3.6      # 上限
# -------------------------------------------------------------

def load_masks(mask_dir, morph_close=MORPH_CLOSE):
    masks = []
    i = 0
    while True:
        p = os.path.join(mask_dir, f"{i}.png")
        if not os.path.exists(p): break
        m = (np.array(Image.open(p).convert("L")) > 127)
        if morph_close and morph_close > 1:
            m = binary_closing(m, square(morph_close))
        masks.append(m)
        i += 1
    return masks

def bbox(seg):
    ys, xs = np.where(seg)
    if len(ys)==0: return None
    y0, x0 = ys.min(), xs.min()
    y1, x1 = ys.max()+1, xs.max()+1
    h, w = y1-y0, x1-x0
    return y0, x0, y1, x1, h, w

def filter_candidates(masks, H, W,
                      area_min=AREA_MIN_PCT, area_max=AREA_MAX_PCT,
                      fill_min=FILL_MIN, fill_max=FILL_MAX,
                      asp_min=ASPECT_MIN, asp_max=ASPECT_MAX):
    S = H*W
    keep = []
    for seg in masks:
        a = int(seg.sum())
        if a == 0: continue
        bb = bbox(seg)
        if bb is None: continue
        y0,x0,y1,x1,h,w = bb
        bbox_area = h*w
        if bbox_area == 0: continue
        fill  = a / bbox_area
        arpct = a / S * 100.0
        asp   = (w+1e-6)/(h+1e-6)
        if (area_min <= arpct <= area_max and
            fill_min <= fill <= fill_max   and
            asp_min  <= asp  <= asp_max):
            cy, cx = (y0+y1)//2, (x0+x1)//2
            keep.append(dict(seg=seg, a=a, y0=y0,x0=x0,y1=y1,x1=x1,
                             h=h,w=w, fill=fill, asp=asp, cy=cy, cx=cx))
    return keep

def cluster_1d(values, thresh):
    if len(values)==0:
        return np.array([], dtype=int), []
    if len(values)==1:
        v = int(values[0])
        return np.array([0], dtype=int), [v]
    X = np.array(values).reshape(-1,1)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, linkage='average')
    labels = model.fit_predict(X)
    centers = [int(np.mean(X[labels==k])) for k in np.unique(labels)]
    order = np.argsort(centers)
    remap = {old:i for i,old in enumerate(order)}
    new_labels = np.array([remap[l] for l in labels], dtype=int)
    centers_sorted = [centers[i] for i in order]
    return new_labels, centers_sorted

def draw_edges_overlay(img, masks, save_path, max_draw=180):
    H,W = img.shape[:2]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(W/160, H/160), dpi=160)
    ax.imshow(img); ax.axis("off")
    # 跳过最大一块（通常是整栋/天空）
    masks_sorted = sorted(masks, key=lambda m: m.sum(), reverse=True)
    for seg in masks_sorted[1:max_draw]:
        contours = measure.find_contours(seg.astype(np.uint8), 0.5)
        for cnt in contours:
            ax.plot(cnt[:,1], cnt[:,0], linewidth=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(pad=0); plt.savefig(save_path, dpi=160)
    plt.close(fig)

def snap_blocks(items, H, W, save_path, row_thr=ROW_MERGE_PX, col_thr=COL_MERGE_PX, palette=PALETTE):
    # 行聚类
    row_labels, _ = cluster_1d([it['cy'] for it in items], row_thr)
    for it, r in zip(items, row_labels):
        it['row'] = int(r)
    rows = (row_labels.max()+1) if len(row_labels) else 0

    canvas = Image.new('RGB', (W,H), (10,30,210))
    draw   = ImageDraw.Draw(canvas)
    color_idx = 0
    total_blocks = 0

    for r in range(rows):
        row_items = [it for it in items if it['row']==r]
        if not row_items: continue
        col_labels, _ = cluster_1d([it['cx'] for it in row_items], col_thr)
        if len(col_labels)==0: continue

        for c in np.unique(col_labels):
            group = [row_items[i] for i in range(len(row_items)) if col_labels[i]==c]
            if not group: continue
            # 用列的中位数宽高更“分明”
            mean_w = int(np.median([g['w'] for g in group]))
            mean_h = int(np.median([g['h'] for g in group]))
            cx = int(np.mean([g['cx'] for g in group]))
            cy = int(np.mean([g['cy'] for g in group]))
            x0, y0 = cx - mean_w//2, cy - mean_h//2
            x1, y1 = x0 + mean_w, y0 + mean_h
            x0 = max(0,x0); y0 = max(0,y0); x1 = min(W,x1); y1 = min(H,y1)
            color = palette[color_idx % len(palette)]
            draw.rectangle([x0,y0,x1,y1], fill=color)
            color_idx += 1
            total_blocks += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    return total_blocks

def process_one(img_path, mask_dir, out_edges, out_blocks, stats_writer):
    img = np.array(Image.open(img_path).convert("RGB"))
    H,W = img.shape[:2]
    masks = load_masks(mask_dir)
    if len(masks)==0:
        return

    # 1) 边缘叠加图
    draw_edges_overlay(img, masks, out_edges)

    # 2) 候选过滤 + 积木化
    cands = filter_candidates(masks, H, W)
    kept  = len(cands)
    blocks = snap_blocks(cands, H, W, out_blocks)

    # 3) 统计
    row_cnt = 0; col_cnt_med = 0
    if kept>0:
        rows_lab, _ = cluster_1d([it['cy'] for it in cands], ROW_MERGE_PX)
        row_cnt = int(rows_lab.max()+1 if len(rows_lab) else 0)
        col_counts = []
        for r in range(row_cnt):
            row_items = [it for it in cands if int(rows_lab[np.where([it2 is it for it2 in cands])[0][0]])==r] \
                        if len(rows_lab)==len(cands) else \
                        [it for it in cands if int(cluster_1d([it['cy'] for it in cands], ROW_MERGE_PX)[0][cands.index(it)])==r]
        # 上面写法太绕，简化为再次聚类并计数：
        rows_lab, _ = cluster_1d([it['cy'] for it in cands], ROW_MERGE_PX)
        for r in range(int(rows_lab.max()+1) if len(rows_lab) else 0):
            row_items = [it for it,lab in zip(cands, rows_lab) if lab==r]
            if not row_items: continue
            cols_lab, _ = cluster_1d([it['cx'] for it in row_items], COL_MERGE_PX)
            col_counts.append(int(cols_lab.max()+1) if len(cols_lab) else 0)
        col_cnt_med = int(np.median(col_counts)) if col_counts else 0

    stats_writer.writerow([os.path.basename(img_path), H, W, len(masks), kept, blocks, row_cnt, col_cnt_med])

def main():
    global ROW_MERGE_PX, COL_MERGE_PX, AREA_MIN_PCT, AREA_MAX_PCT, FILL_MIN, FILL_MAX, ASPECT_MIN, ASPECT_MAX

    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pattern", default="jpg")

    # 这些默认值可以直接用上面的全局常量（因为 global 已在最前）
    ap.add_argument("--row_merge", type=int, default=ROW_MERGE_PX)
    ap.add_argument("--col_merge", type=int, default=COL_MERGE_PX)
    ap.add_argument("--area_min",  type=float, default=AREA_MIN_PCT)
    ap.add_argument("--area_max",  type=float, default=AREA_MAX_PCT)
    ap.add_argument("--fill_min",  type=float, default=FILL_MIN)
    ap.add_argument("--fill_max",  type=float, default=FILL_MAX)
    ap.add_argument("--asp_min",   type=float, default=ASPECT_MIN)
    ap.add_argument("--asp_max",   type=float, default=ASPECT_MAX)
    args = ap.parse_args()

    # 把命令行值回写到全局（供下面函数使用）
    ROW_MERGE_PX, COL_MERGE_PX = args.row_merge, args.col_merge
    AREA_MIN_PCT, AREA_MAX_PCT = args.area_min, args.area_max
    FILL_MIN, FILL_MAX         = args.fill_min, args.fill_max
    ASPECT_MIN, ASPECT_MAX     = args.asp_min,  args.asp_max

    os.makedirs(args.out_root, exist_ok=True)
    csv_path = os.path.join(args.out_root, "stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image","H","W","masks_total","candidates_kept","blocks_drawn","rows","median_cols_per_row"])

        # 遍历 mask_root 下的所有 cmp_* 文件夹
        subdirs = sorted([d for d in os.listdir(args.mask_root) if os.path.isdir(os.path.join(args.mask_root,d))])
        for sd in subdirs:
            mask_dir = os.path.join(args.mask_root, sd)
            # 对应原图：img_dir 下的 sd.jpg / .png
            stem = sd
            img_path = None
            for ext in [args.pattern, "png", "jpeg", "jpg"]:
                p = os.path.join(args.img_dir, f"{stem}.{ext}")
                if os.path.exists(p):
                    img_path = p; break
            if img_path is None:
                print(f"[skip] image not found for {sd}")
                continue

            out_edges  = os.path.join(args.out_root, "overlay_edges", f"{stem}_overlay_edges.png")
            out_blocks = os.path.join(args.out_root, "blocks", f"{stem}_blocks.png")

            try:
                process_one(img_path, mask_dir, out_edges, out_blocks, writer)
                print("done:", stem)
            except Exception as e:
                print("error:", stem, e)
                continue

    print("All done. See:", args.out_root)

if __name__ == "__main__":
    main()
