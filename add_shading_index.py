import os, argparse, numpy as np, pandas as pd
from PIL import Image
from glob import glob

# 判定“水平带”的阈值（可调）
AR_MIN = 4.0    # 宽/高 > 4 视为水平带
FILL_MAX = 0.8  # 填充过满的排除（避免把整片墙当带）
AREA_MIN = 0.001  # 占整图比例的最小面积

def load_masks(d):
    ps=sorted(glob(os.path.join(d,'*.png')),key=lambda p:int(os.path.basename(p).split('.')[0]))
    return [(np.array(Image.open(p).convert('L'))>127) for p in ps]

def bbox(m):
    ys,xs=np.where(m)
    if ys.size==0: return None
    y0,x0=ys.min(),xs.min(); y1,x1=ys.max()+1,xs.max()+1
    return y0,x0,y1,x1,(y1-y0),(x1-x0)

def shading_index(mask_dir):
    ms=load_masks(mask_dir)
    if not ms: return np.nan
    big=max(ms,key=lambda m:m.sum())
    H,W=big.shape; S=H*W
    shade=0.0
    for seg in ms:
        a=int(seg.sum()); 
        if a<S*AREA_MIN: continue
        b=bbox(seg); 
        if b is None: continue
        y0,x0,y1,x1,h,w=b
        bb=h*w or 1
        fill=a/bb
        ar=(w+1e-6)/(h+1e-6)
        if ar>AR_MIN and fill<FILL_MAX:
            shade+=a
    return float(shade)/float(big.sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stats_in',required=True)
    ap.add_argument('--mask_root',required=True)
    ap.add_argument('--stats_out',required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.stats_in)
    vals=[]
    for _,r in df.iterrows():
        stem=os.path.splitext(str(r['image']))[0]
        d=os.path.join(args.mask_root,stem)
        vals.append(shading_index(d) if os.path.isdir(d) else np.nan)
    df['shading_idx']=vals
    df.to_csv(args.stats_out,index=False)
    print('saved:',args.stats_out)
if __name__=='__main__': main()
