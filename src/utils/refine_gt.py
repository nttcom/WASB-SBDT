from tqdm import tqdm
import numpy as np
import cv2

from utils import read_image, compute_l2_dist_mat
from utils import Center

def load_refine_gt_npz(npz_path, num_clusters=None, margin=None):
    tmp          = np.load(npz_path)
    centroids    = tmp['centroids']
    patches      = tmp['patches']
    num_clusters = tmp['num_clusters']
    margin       = tmp['margin']
    cxys         = tmp['cxys']
    sizes        = tmp['sizes']
    if num_clusters is not None:
        if num_clusters!=tmp['num_clusters']:
            raise ValueError('saved data come from different parameters')
    if margin is not None:
        if margin!=tmp['margin']:
            raise ValueError('saved data come from different parameters')
    return tmp

def refine_gt_clip_tennis(ball_xyvs, 
                          frame_dir, 
                          frame_names, 
                          npz_path,
                          ratio = 0.8
):

    npz_data  = load_refine_gt_npz(npz_path)
    margin    = npz_data['margin']
    centroids = npz_data['centroids_filtered']
    cxys      = npz_data['cxys']
    sizes     = npz_data['sizes']
    v2c_dists = npz_data['v2c_dists']
    dist_thresh = np.sort(v2c_dists)[int(v2c_dists.shape[0]*0.8)]
    
    ball_xyvs_new = {}
    cnt_refined   = 0
    for ind, ball_xyv in tqdm( ball_xyvs.items() ):
        xy_gt, visi_gt = ball_xyv['center'].xy, ball_xyv['center'].is_visible

        frame_path = ball_xyv['frame_path']
        im          = read_image(frame_path)
        im          = np.asarray(im)
        im_h,im_w,_ = im.shape

        if not visi_gt:
            ball_xyvs_new[ind] = ball_xyvs[ind]
            continue

        x,y = xy_gt
        if np.isnan(x) or np.isnan(y):
            ball_xyvs_new[ind] = ball_xyvs[ind]
            continue

        min_x, max_x = max(int(x)-margin, 0), min(int(x)+margin+1, im_w-1)
        min_y, max_y = max(int(y)-margin, 0), min(int(y)+margin+1, im_h-1)
        im_crop      = im[min_y:max_y, min_x:max_x]

        if im_crop.shape[0]*im_crop.shape[1]!=(margin*2+1)**2:
            ball_xyvs_new[ind] = ball_xyvs[ind]
            continue

        im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_crop = im_crop.reshape(-1).astype(np.float64) / 255.

        im_crop = im_crop[np.newaxis, :]

        dists = compute_l2_dist_mat(im_crop, centroids)
        dists = np.squeeze(dists)
        cind  = np.argmin(dists)
        if dists[cind] > dist_thresh:
            ball_xyvs_new[ind] = ball_xyvs[ind]
            continue

        x_new = min_x + cxys[cind][0]
        y_new = min_y + cxys[cind][1]
        size  = sizes[cind]
        #print(xy_gt, x_new, y_new)
        ball_xyvs_new[ind] = { 'center': Center(x=x_new, y=y_new, is_visible=visi_gt, r=size),
                               'file_name': ball_xyv['file_name'],
                               'frame_path': ball_xyv['frame_path'],
                               }
        cnt_refined += 1

    print('{}/{} refined'.format(cnt_refined, len(ball_xyvs))) 
    return ball_xyvs_new

