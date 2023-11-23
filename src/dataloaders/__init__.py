import os
import os.path as osp
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import dataloaders.img_transforms as T
import dataloaders.seq_transforms as ST

from .dataset_loader import ImageDataset, read_image, get_transform
from .samplers import select_sampler, RandomSampler
from datasets import select_dataset

def build_img_transforms(cfg):
    transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test

def build_seq_transforms(cfg):
    transform_train = ST.SeqTransformCompose([
        ST.RandomHorizontalFlipping(cfg['transform']['train']['horizontal_flip']['p']),
        ST.RandomCropping(p=cfg['transform']['train']['crop']['p'], max_rescale=cfg['transform']['train']['crop']['max_rescale']),
    ])
    transform_test = None
    return transform_train, transform_test

def build_dataloader( 
    cfg: DictConfig,
):
    dataset = select_dataset(cfg)

    transform_train, transform_test         = build_img_transforms(cfg)
    seq_transform_train, seq_transform_test = build_seq_transforms(cfg)
    
    input_wh  = (cfg['model']['inp_width'], cfg['model']['inp_height'])
    output_wh = (cfg['model']['out_width'], cfg['model']['out_height'])

    train_sampler, test_sampler, train_clip_samplers, test_clip_samplers = select_sampler(cfg['dataloader']['sampler'], dataset)
    
    model_name = cfg['model']['name']

    fp1_fname = cfg['runner']['fp1_filename']
    if fp1_fname is not None:
        fp1_fpath = osp.join(cfg['output_dir'], fp1_fname)
    else:
        fp1_fpath = None

    train_clip_datasets = {}
    test_clip_datasets  = {}
    if model_name in ['tracknetv2', 'hrnet', 'monotrack', 'restracknetv2', 'deepball', 'ballseg']:
        train_dataset = ImageDataset(cfg, 
                                     dataset=dataset.train, 
                                     input_wh=input_wh, 
                                     output_wh=output_wh, 
                                     transform=transform_train, 
                                     seq_transform=seq_transform_train,
                                     fp1_fpath=fp1_fpath,
                                     )
        test_dataset  = ImageDataset(cfg, 
                                     dataset=dataset.test, 
                                     input_wh=input_wh, 
                                     output_wh=output_wh, 
                                     transform=transform_test, 
                                     seq_transform=seq_transform_test, 
                                     is_train=False,
                                     )

        for key, clip in dataset.train_clips.items():
            clip_dataset  = ImageDataset(cfg, 
                                         dataset=clip, 
                                         input_wh=input_wh, 
                                         output_wh=output_wh, 
                                         transform=transform_test, 
                                         is_train=False,
                                         )
            train_clip_datasets[key] = clip_dataset

        for key, clip in dataset.test_clips.items():
            clip_dataset  = ImageDataset(cfg, 
                                         dataset=clip, 
                                         input_wh=input_wh, 
                                         output_wh=output_wh, 
                                         transform=transform_test, 
                                         is_train=False,
                                         )
            test_clip_datasets[key] = clip_dataset 
    else:
        raise ValueError('unknwon model_name : {}'.format(model_name))


    train_loader = DataLoader(dataset=train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=cfg['dataloader']['train_num_workers'],
                              pin_memory=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_sampler=test_sampler,
                             num_workers=cfg['dataloader']['test_num_workers'],
                             pin_memory=False)

    train_clip_loaders_and_gts = {}
    for key, clip_dataset in train_clip_datasets.items():
        clip_loader = DataLoader(dataset=clip_dataset,
                                 batch_sampler=train_clip_samplers[key],
                                 num_workers=cfg['dataloader']['inference_video_num_workers'],
                                 pin_memory=False)
        train_clip_loaders_and_gts[key] = {'clip_loader': clip_loader, 'clip_gt': dataset.train_clip_gts[key]}

    test_clip_loaders_and_gts = {}
    for key, clip_dataset in test_clip_datasets.items():
        clip_loader = DataLoader(dataset=clip_dataset,
                                 batch_sampler=test_clip_samplers[key],
                                 num_workers=cfg['dataloader']['inference_video_num_workers'],
                                 pin_memory=False)
        test_clip_loaders_and_gts[key] = {'clip_loader': clip_loader, 'clip_gt': dataset.test_clip_gts[key]}

    return train_loader, test_loader, train_clip_loaders_and_gts, test_clip_loaders_and_gts

