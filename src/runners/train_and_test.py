import os
import os.path as osp
import shutil
import time
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn

from models import build_model
from dataloaders import build_dataloader
from losses import build_loss_criteria
from optimizers import build_optimizer_and_scheduler
from utils import save_checkpoint, set_seed, mkdir_if_missing, count_params, AverageMeter
from .inference_videos import VideosInferenceRunner
from .base import BaseRunner
from .runner_utils import train_epoch, test_epoch

log = logging.getLogger(__name__)


def update_fp1_example(epoch,
                       model, 
                       vi_runner,
                       fp1_fpath,
):
    vi_results = vi_runner.run(model=model)
    print(vi_results['fp1_im_list_dict'])
    print(fp1_fpath)
    fp1_im_list_dict = vi_results['fp1_im_list_dict']
    with open(fp1_fpath, 'w') as f:
        for key, im_list in fp1_im_list_dict.items():
            for path in im_list:
                f.write('{}\n'.format(path))
    fp1_fpath_current = osp.splitext(fp1_fpath)[0] + '_{}.txt'.format(epoch)
    shutil.copyfile(fp1_fpath, fp1_fpath_current)

class Trainer(BaseRunner):
    def __init__(self,
                 cfg: DictConfig,
    ):
        super().__init__(cfg)

        seed = self._cfg['seed']
        set_seed(seed)
        
        self._device = cfg['runner']['device']
        if self._device!='cuda':
            assert 0, 'device=cpu not supported'
        if not torch.cuda.is_available():
            assert 0, 'GPU NOT available'
        self._gpus  = self._cfg['runner']['gpus']

        self._max_epoch  = cfg['runner']['max_epochs']

        self._run_test          = cfg['runner']['test']['run']
        if self._run_test:
            assert 0, 'not yet (2023.4.23)'
        self._start_test        = cfg['runner']['test']['epoch_start']
        self._step_test         = cfg['runner']['test']['epoch_step']
        self._test_before_train = cfg['runner']['test']['run_before_train']
        self._test_after_train_with_best = cfg['runner']['test']['run_after_train_with_best']
        if self._step_test < 1:
            raise ValueError('step_test >= 1, but is set to {}'.format(self._step_test))
        #print(self._run_test, self._start_test, self._step_test, self._test_before_train)

        self._run_inference_video          = cfg['runner']['inference_video']['run']
        self._start_inference_video        = cfg['runner']['inference_video']['epoch_start']
        self._step_inference_video         = cfg['runner']['inference_video']['epoch_step']
        self._inference_video_before_train = cfg['runner']['inference_video']['run_before_train']
        self._inference_video_after_train_with_best = cfg['runner']['inference_video']['run_after_train_with_best']
        if self._step_inference_video < 1:
            raise ValueError('step_inference_video >= 1, but is set to {}'.format(self._step_inference_video))
        #print(self._run_inference_video, self._start_inference_video, self._step_inference_video, self._inference_video_before_train)

        self._model                     = build_model(cfg)
        log.info(self._model)
        log.info('# model params: (trainable) {}, (whole) {}'.format(count_params(self._model), count_params(self._model, only_trainable=False)))
        
        self._train_loader, self._test_loader, self._train_clip_loaders_and_gts, self._test_clip_loaders_and_gts = build_dataloader(cfg)
        self._loss_criteria = build_loss_criteria(cfg)
        self._optimizer, self._scheduler = build_optimizer_and_scheduler(cfg, list(self._model.parameters())+list(self._loss_criteria.parameters()) )

        self._model = self._model.to(self._device)
        self._model = nn.DataParallel(self._model, device_ids=self._gpus)
        
        self._loss_criteria = self._loss_criteria.to(self._device)
        
        if self._run_inference_video:
            self._vi_runner = VideosInferenceRunner(self._cfg,
                                                    clip_loaders_and_gts=self._test_clip_loaders_and_gts,
                                                    vis_result=False, vis_hm=False
                                                    )

        self._find_fp1_epochs = cfg['runner']['find_fp1_epochs']
        if len(self._find_fp1_epochs) > 0:
            fp1_fname = cfg['runner']['fp1_filename']
            if fp1_fname is None:
                raise ValueError('fp_filename is not defined')
            self._fp1_fpath = osp.join(self._output_dir, fp1_fname)
            self._train_vi_runner = VideosInferenceRunner(self._cfg,
                                                    clip_loaders_and_gts=self._train_clip_loaders_and_gts,
                                                    vis_result=False, vis_hm=False
                                                    )

        self._best_model_name = cfg['runner']['best_model_name']

    def run(self):

        if self._test_before_train:
            test_epoch(0, model, test_loader, loss_criteria, self._device, cfg)
        if self._inference_video_before_train:  
            self._vi_runner.run(model=self._model)
        
        best_loss  = np.Inf
        best_f1acc = -np.Inf
        #is_best    = False
        for epoch in range(self._max_epoch):
            log.info('(TRAIN) Epoch {}, lr: {}'.format(epoch+1, self._scheduler.get_last_lr()))

            if epoch+1 in self._find_fp1_epochs:
                log.info('find fp1 examples @ Epoch {}'.format(epoch+1))
                update_fp1_example(epoch+1, self._model, self._train_vi_runner, self._fp1_fpath)

            train_results = train_epoch(epoch+1, 
                                        self._model, 
                                        self._train_loader, 
                                        self._loss_criteria, 
                                        self._optimizer, 
                                        self._device
                            )

            is_best = False

            test_results = {'loss': None, 'prec': None, 'recall': None, 'f1': None, 'accuracy': None}
            if (epoch+1 > self._start_test) and ( (epoch+1) % self._step_test==0 or (epoch+1) == self._max_epoch):
                if self._run_test:
                    log.info('(TEST) @ Epoch {}'.format(epoch+1))
                    torch.cuda.empty_cache()
                    test_results = test_epoch(epoch+1, self._model, self._test_loader, self._loss_criteria, self._device, self._cfg)
                    torch.cuda.empty_cache()

            vi_results = {'prec': None, 'recall': None, 'f1': None, 'accuracy': None}
            if (epoch+1 > self._start_inference_video) and ( (epoch+1) % self._step_inference_video==0 or (epoch+1) == self._max_epoch):
                if self._run_inference_video:
                    log.info('(INFERENCE_VIDEO) @ Epoch {}'.format(epoch+1))
                    torch.cuda.empty_cache()
                    vi_results = self._vi_runner.run(model=self._model)
                    torch.cuda.empty_cache()
                    is_best = vi_results['f1']+vi_results['accuracy'] > best_f1acc
                    if is_best:
                        log.info('best f1 ({}) + acc ({}) @ epoch {}'.format(vi_results['f1'], vi_results['accuracy'], epoch+1))
                        best_f1acc = vi_results['f1'] + vi_results['accuracy']

            model_state_dict = self._model.module.state_dict()
            save_checkpoint({'model_state_dict':model_state_dict, 
                             'train_loss': train_results['loss'],
                             'test_loss': test_results['loss'],
                             'test_prec': test_results['prec'],
                             'test_recall': test_results['recall'],
                             'test_f1': test_results['f1'],
                             'test_accuracy': test_results['accuracy'],
                             'video_inference_prec': vi_results['prec'],
                             'video_inference_recall': vi_results['recall'],
                             'video_inference_f1': vi_results['f1'],
                             'video_inference_accuracy': vi_results['accuracy'],
                             'epoch': epoch+1
                             }, 
                             is_best, 
                             model_path=osp.join(self._output_dir, 'checkpoint_ep{}.pth.tar'.format(epoch+1)),
                             best_model_name=self._best_model_name,
                             )

            self._scheduler.step()

        if self._inference_video_after_train_with_best:
            self._vi_runner.run(model_dir=self._output_dir)

