import os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure

img_path = r"F:\ucl-term1\facade\cmp_b0001.jpg"
mask_dir = r"F:\ucl-term1\segment-anything-main\out_facade\cmp_b0001"
save_to  = r"F:\ucl-term1\segment-anything-main\out_facade\cmp_b0001_overlay_edges.png"

img = np.array(Image.open(img_path).convert("RGB"))
H, W = img.shape[:2]

# 读取所有掩膜并按面积排序
masks = []
i = 0
while True:
    p = os.path.join(mask_dir, f"{i}.png")
    if not os.path.exists(p): break
    m = (np.array(Image.open(p).convert("L")) > 127)
    masks.append((i, m, int(m.sum())))
    i += 1
masks.sort(key=lambda x: x[2], reverse=True)

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img)
ax.axis("off")

# 跳过面积最大的那块（通常是整栋建筑/天空）
for idx, m, area in masks[1:150]:    # 只画前150个
    # 提取轮廓线
    contours = measure.find_contours(m.astype(np.uint8), 0.5)
    for cnt in contours:
        ax.plot(cnt[:,1], cnt[:,0], linewidth=1)

plt.tight_layout()
plt.savefig(save_to, dpi=200)
print("saved:", save_to)
