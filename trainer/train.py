import torch

from tools import BaseRunner
from utils import get_root_logger, save_checkpoint, mkdir_or_exist, tensor_to_device
import time
import numpy as np
import os.path as osp
from tqdm import tqdm
from utils.metrics.rotate_metrics import combine_predicts_gt
from utils.metrics import RotateDetEval



class Train(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(Train, self).__init__(*args, **kwargs)
        if self.network_type == "rotate_detection":
            self.eval_method = RotateDetEval()

    def _train_epoch(self, epoch):
        self.logger = get_root_logger(log_level='INFO')
        all_losses = []
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        lr = self.optimizer.param_groups[0]['lr']
        for count, data in enumerate(self.train_dataloader):
            if count >= len(self.train_dataloader):
                break
            self.global_step += 1

            tensor_to_device(data, self.device)

            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            batch = _img.shape[0]
            predict, losses = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
            losses = losses["loss"]
            losses.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses = losses.detach().cpu().numpy()
            all_losses.append(losses)
            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger.info('[%d/%d], [%d/%d], training step: %d, running loss: %f, time: %d ms' % (
                    epoch, self.epochs, (count+1)*batch, len(self.train_dataloader.dataset), self.global_step, np.array(all_losses).mean(), batch_time * 1000))
                batch_start = time.time()
            if self.save_train_metrics_log:
                pass
            if self.save_train_predict_fn:
                pass

        return {'train_loss': sum(all_losses) / len(self.train_dataloader), 'lr': lr,
                'time': time.time() - epoch_start, 'epoch': epoch}

    def _after_epoch(self, results):
        self.logger.info('[%d/%d], train_loss: %f, time: %d ms, lr: %5d' % (
            results['epoch'], self.epochs, results['train_loss'],
            results['time'] * 1000, results['lr']))
        model_save_dir = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                         f'{self.time_str}/checkpoints'
        net_save_path_best = osp.join(model_save_dir, 'model_best.pth')
        net_save_path_loss_best = osp.join(model_save_dir, f'model_best_loss.pth')
        assert self.val_dataloader is not None, "no val data in the dataset"
        precision, recall, mAP = self._eval(results['epoch'])
        if mAP >= self.metrics['mAP']:
            self.metrics['train_loss'] = results['train_loss']
            self.metrics['mAP'] = mAP
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['best_model_epoch'] = results['epoch']
            save_checkpoint(self.model, net_save_path_best)
        elif results['train_loss'] <= self.metrics['train_loss']:
            self.metrics['train_loss'] = results['train_loss']
            self.metrics['best_model_epoch'] = results['epoch']
            save_checkpoint(self.model, net_save_path_loss_best)
        else:
            pass
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.logger(best_str)
        self.logger('finish training.')

    def _eval(self, epoch):
        self.model.eval()
        final_collection = []
        total_frame = 0.0
        total_time = 0.0
        for i, data in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc='test model'):
            with torch.no_grad():
                tensor_to_device(data, self.device)
            start_time = time.time()
            _img, ground_truth = data['images_collect']['img'], data['ground_truth']
            total_time += time.time() - start_time
            cur_batch = _img.shape[0]
            total_frame += cur_batch
            predicts = self.model(_img)
            for i in range(cur_batch):
                predict_gt_collection = combine_predicts_gt(predicts[i], data['images_collect']['img_metas'][i],
                                                            [ground_truth[key][i] for key in ground_truth])
                final_collection.append(predict_gt_collection)
        metric = self.eval_method(final_collection)
        self.logger.info(f'FPS:{total_frame / total_time}')
        return metric

