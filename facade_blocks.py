import os, numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering
from skimage.morphology import binary_closing, square

# ====== 参数区（按图可调）======
IMG_PATH  = r"F:\ucl-term1\facade\cmp_b0001.jpg"               # 原图，用来取尺寸
MASK_DIR  = r"F:\ucl-term1\segment-anything-main\out_facade\cmp_b0001"  # 该图的掩膜文件夹
SAVE_PATH = r"F:\ucl-term1\segment-anything-main\out_facade\cmp_b0001_blocks.png"

# 过滤窗口类候选（经验阈值，按需要微调）
AREA_MIN_PCT  = 0.03    # mask面积下限（% of image）
AREA_MAX_PCT  = 8.0     # mask面积上限
FILL_MIN      = 0.30    # 填充比=area/bbox_area
FILL_MAX      = 0.92
ASPECT_MIN    = 0.33    # 宽高比 w/h
ASPECT_MAX    = 3.20

# 网格对齐参数
ROW_MERGE_PX  = 18      # 行聚类的垂直阈值（像素）
COL_MERGE_PX  = 18      # 每一行内列聚类的水平阈值（像素）
MORPH_CLOSE   = 3       # 先对二值mask做一次闭运算以合并碎片（像素窗口大小）
PALETTE       = [(26,54,255),(17,132,255),(0,191,255),(0,230,171),
                 (148,231,0),(227,211,0),(255,153,0),(255,66,0)]  # 8色调色板
# =================================

def load_masks(mask_dir):
    masks = []
    i = 0
    while True:
        p = os.path.join(mask_dir, f"{i}.png")
        if not os.path.exists(p): break
        m = np.array(Image.open(p).convert("L")) > 127
        if MORPH_CLOSE > 1:
            m = binary_closing(m, square(MORPH_CLOSE))
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

def filter_window_like(masks, H, W):
    S = H*W
    keep = []
    for seg in masks:
        a = int(seg.sum())
        if a == 0: continue
        y0,x0,y1,x1,h,w = bbox(seg)
        bbox_area = h*w
        if bbox_area == 0: continue
        fill  = a / bbox_area
        arpct = a / S * 100.0
        asp   = (w+1e-6)/(h+1e-6)
        if (AREA_MIN_PCT <= arpct <= AREA_MAX_PCT and
            FILL_MIN <= fill <= FILL_MAX and
            ASPECT_MIN <= asp <= ASPECT_MAX):
            cy, cx = (y0+y1)//2, (x0+x1)//2
            keep.append(dict(seg=seg, a=a, y0=y0,x0=x0,y1=y1,x1=x1,
                             h=h,w=w, fill=fill, asp=asp, cy=cy, cx=cx))
    # 去掉最大那类（通常是整栋或大背景）——已经通过面积阈值基本过滤
    return keep

def cluster_1d(values, thresh):
    """对一维坐标做层次聚类；当样本 < 2 时直接返回。"""
    if len(values) == 0:
        return np.array([], dtype=int), []
    if len(values) == 1:
        v = int(values[0])
        return np.array([0], dtype=int), [v]

    X = np.array(values).reshape(-1, 1)
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thresh, linkage='average'
    )
    labels = model.fit_predict(X)

    centers = []
    for k in np.unique(labels):
        centers.append(int(np.mean(X[labels == k])))

    order = np.argsort(centers)
    remap = {old: i for i, old in enumerate(order)}
    new_labels = np.array([remap[l] for l in labels], dtype=int)
    centers_sorted = [centers[i] for i in order]
    return new_labels, centers_sorted

from PIL import Image, ImageDraw

def snap_and_draw(items, H, W, save_path=None, bg=(10,30,210)):
    # 行聚类
    row_labels, row_centers = cluster_1d([it['cy'] for it in items], ROW_MERGE_PX)
    for it, r in zip(items, row_labels):
        it['row'] = int(r)
    rows = (row_labels.max() + 1) if len(row_labels) else 0

    canvas = Image.new('RGB', (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    color_idx = 0
    for r in range(rows):
        row_items = [it for it in items if it['row'] == r]
        if not row_items:
            continue

        col_labels, col_centers = cluster_1d([it['cx'] for it in row_items], COL_MERGE_PX)

        # 避免空切片
        unique_cols = np.unique(col_labels) if len(col_labels) else []
        for c in unique_cols:
            col_group = [row_items[i] for i in range(len(row_items)) if col_labels[i] == c]
            if not col_group:
                continue
            mean_w = int(np.mean([g['w'] for g in col_group]))
            mean_h = int(np.mean([g['h'] for g in col_group]))
            cx = int(np.mean([g['cx'] for g in col_group]))
            cy = int(np.mean([g['cy'] for g in col_group]))
            x0, y0 = cx - mean_w // 2, cy - mean_h // 2
            x1, y1 = x0 + mean_w, y0 + mean_h
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            color = PALETTE[color_idx % len(PALETTE)]
            draw.rectangle([x0, y0, x1, y1], fill=color)
            color_idx += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
    return canvas

# ========== 主流程 ==========
img = Image.open(IMG_PATH).convert("RGB")
W, H = img.size
masks = load_masks(MASK_DIR)
cands = filter_window_like(masks, H, W)
canvas = snap_and_draw(cands, H, W)
canvas.save(SAVE_PATH)
print(f"kept {len(cands)} parts -> {SAVE_PATH}")
