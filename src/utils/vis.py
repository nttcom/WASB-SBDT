import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import cv2

from utils import Center

def draw_frame(img_or_path,
               center: Center,
               color: Tuple,
               radius : int = 5,
               thickness : int = -1,
    ):
        if osp.isfile(img_or_path):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        xy   = center.xy
        visi = center.is_visible
        if visi:
            x, y = xy
            x, y = int(x), int(y)
            img  = cv2.circle(img, (x,y), radius, color, thickness=thickness)
        
        return img
        
def gen_video(video_path, 
              vis_dir, 
              resize=1.0, 
              fps=30.0, 
              fourcc='mp4v'
):

    fnames = os.listdir(vis_dir)
    fnames.sort()
    h,w,_   = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w*resize), int(h*resize))
    fourcc  = cv2.VideoWriter_fourcc(*fourcc)
    out     = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im      = cv2.imread(im_path)
        im = cv2.resize(im, None, fx=resize, fy=resize)
        out.write(im)

