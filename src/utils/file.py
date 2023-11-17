import os.path as osp
import pandas as pd
import numpy as np

from utils import Center

def load_csv_tennis(csv_path, visible_flags, frame_dir=None):
    df = pd.read_csv(csv_path)
    fnames, visis, xs, ys = df['file name'].tolist(), df['visibility'].tolist(), df['x-coordinate'].tolist(), df['y-coordinate'].tolist()
    xyvs = {}
    for fname, visi, x, y in zip(fnames, visis, xs, ys):
        fid = int(osp.splitext(fname)[0])
        if frame_dir is not None:
            frame_path = osp.join(frame_dir, fname)
        else:
            frame_path = None

        if fid in xyvs.keys():
            raise KeyError('fid {} already exists'.format(fid ))

    xyvs = {}
    for fname, visi, x, y in zip(fnames, visis, xs, ys):
        fid = int(osp.splitext(fname)[0])
        if frame_dir is not None:
            frame_path = osp.join(frame_dir, fname)
        else:
            frame_path = None

        if fid in xyvs.keys():
            raise KeyError('fid {} already exists'.format(fid ))
        
        if np.isnan(x) or np.isnan(y):
            if (int(visi) in visible_flags):
                print(visible_flags)
                print(fname)
                print(visi, x, y)
                print(int(visi))
                quit()

        xyvs[fid] = {'center': Center(x=float(x),
                                      y=float(y),
                                      is_visible=True if int(visi) in visible_flags else False,
                               ),
                     'file_name': fname,
                     'frame_path': frame_path
                     }

    return xyvs

