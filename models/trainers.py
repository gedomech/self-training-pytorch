# coding=utf-8
import copy
import logging
import sys
import os
import warnings
from abc import ABC

from absl import flags
from tensorboardX import SummaryWriter

import utils.mask_gene
from loss.loss import get_citerion
from utils.helpers import *
from utils.logger import config_logger

logger = logging.getLogger(__name__)
logger.parent = None
sys.path.extend([os.path.dirname(os.getcwd())])

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(ABC):
    def __init__(self, torchnet) -> None:
        super().__init__()
        self.torchnet = torchnet
        self.name = 'base'
        self.hparam = None
        self.lrscheduler = None
        self.dataloader = None
        self.criterion = None
        self.dataloader = None

    def _train(self, **kwargs):

        pass

    def set_writer(self, writer):
        self.writer = writer

    def start_training(self, savedir):
        logger.info(self.name + '  Training starts:')
        for epoch in range(self.hparam['max_epoch']):
            self.evaluate(epoch, mode='train', savedir=savedir)
            self.evaluate(epoch, mode='eval', savedir=savedir)

            self._train(self.dataloader)
            self.lrscheduler.step()

    def evaluate(self, epoch, mode='eval', savedir=None):
        with torch.no_grad():
            metrics = {}
            dice = self._evaluate(self.dataloader['labeled'], mode)
            logger.info('at epoch: {:3d}, under {} mode, labeled_data dice: {:.3f} '.format(epoch, mode, dice))
            # self.writer.add_scalar('%s/labeled' % self.name, dice, epoch)
            metrics['%s/labeled' % self.name] = dice
            dice = self._evaluate(self.dataloader['unlabeled'], mode)
            logger.info('at epoch: {:3d}, under {} mode, unlabeled_data dice: {:.3f} '.format(epoch, mode, dice))
            # self.writer.add_scalar('%s/unlabeled' % self.name, dice, epoch)
            metrics['%s/unlabeled' % self.name] = dice
            dice = self._evaluate(self.dataloader['val'], mode)
            logger.info('at epoch: {:3d}, under {} mode, val_data dice: {:.3f} '.format(epoch, mode, dice))
            # self.writer.add_scalar('%s/val' % self.name, dice, epoch)
            metrics['%s/val' % self.name] = dice
        if mode == 'eval':
            self.writer.add_scalars(self.name, metrics, epoch)
            self.checkpoint(dice, epoch, savedir)

    def _evaluate(self, dataloader, mode='eval'):
        assert mode in ('eval', 'train')
        dice_meter = AverageValueMeter()
        if mode == 'eval':
            self.torchnet.eval()
        else:
            self.torchnet.train()

        with torch.no_grad():
            for i, (img, gt, _) in enumerate(dataloader):
                img, gt = img.to(device), gt.to(device)
                pred_logit = self.torchnet(img)
                pred_mask = pred2segmentation(pred_logit)
                dice_meter.add(dice_loss(pred_mask, gt))
        if mode == 'eval':
            self.torchnet.train()
        assert self.torchnet.training == True
        return dice_meter.value()[0]

    @property
    def save_dict(self):
        return self.torchnet.state_dict()

    @classmethod
    def _rm_alias(cls, hparam):
        new_hparam = {}
        for k, v in hparam.items():
            if k.find(cls.alias) >= 0:
                new_hparam[k.replace(cls.alias, '')] = v
        return new_hparam

    def checkpoint(self, dice, epoch, name=None):
        try:
            getattr(self, 'best_dice')
        except:
            self.best_dice = -1

        ## save this checkpoint as last.pth
        dict2save = {}
        dict2save['epoch'] = epoch
        dict2save['dice'] = dice
        dict2save['model'] = self.save_dict
        if name is None:
            torch.save(dict2save, 'last.pth')
        else:
            torch.save(dict2save, name + '/last.pth')

        if dice > self.best_dice:
            self.best_dice = dice
            dict2save = dict()
            dict2save['epoch'] = epoch
            dict2save['dice'] = dice
            dict2save['model'] = self.save_dict
            if name is None:
                torch.save(dict2save, 'best.pth')
            else:
                torch.save(dict2save, name + '/best.pth')
        else:
            return

    def load_checkpoint(self, path):
        model = torch.load(path, map_location=lambda storage, loc: storage)
        logger.info('Saved_epoch: {}, Dice: {:3f}'.format(model['epoch'], model['dice']))
        self.torchnet.load_state_dict(model['model'])


class FullysupervisedTrainer(Trainer):
    alias = 'full_train__'
    lrscheduler_keys = ['milestones', 'gamma']
    optim_keys = ['lr', 'weight_decay']
    criterion_keys = ['weight']

    @classmethod
    def set_flag(cls):
        flags.DEFINE_integer(cls.alias + 'max_epoch', default=200, help='max_epoch for full training')
        flags.DEFINE_multi_integer(cls.alias + 'milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
                                   help='milestones for full training')
        flags.DEFINE_float(cls.alias + 'gamma', default=0.5, help='gamma for lr_scheduler in full training')
        flags.DEFINE_float(cls.alias + 'lr', default=0.001, help='lr for full training')
        flags.DEFINE_float(cls.alias + 'weight_decay', default=0, help='weight_decay for full training')
        flags.DEFINE_multi_float(cls.alias + 'weight', default=[1, 1], help='weight balance for CE for full training')
        flags.DEFINE_string(cls.alias + 'loss_name', default='crossentropy', help='criterion used in the full training')
        flags.DEFINE_string(cls.alias + 'optim_name', default='Adam', help='optimzer used in the full training')
        flags.DEFINE_string(cls.alias + 'optim_option', default='{}', help='optimzer used in the full training')
        flags.DEFINE_string(cls.alias + 'scheduler', default='MultiStepLR', help='scheduler used in the full training')

    def __init__(self, torchnet, dataloader, hparam) -> None:
        super().__init__(torchnet)
        self.name = 'Fully_Supervised_Training'
        self.dataloader = dataloader
        self.hparam = copy.deepcopy(self._rm_alias(hparam))
        optim_hparam = extract_from_big_dict(self.hparam, FullysupervisedTrainer.optim_keys)
        optim_hparam.update(**eval(self.hparam['optim_option']))
        self.optim = getattr(torch.optim, self.hparam['optim_name'])(self.torchnet.parameters(), **optim_hparam)
        lrschduler_hparam = extract_from_big_dict(self.hparam, FullysupervisedTrainer.lrscheduler_keys)
        self.lrscheduler = getattr(torch.optim.lr_scheduler, self.hparam['scheduler'])(self.optim, **lrschduler_hparam)
        criterion_hparam = extract_from_big_dict(self.hparam, FullysupervisedTrainer.criterion_keys)
        self.criterion = get_citerion(self.hparam['loss_name'], **criterion_hparam)
        self.criterion.to(device)

    def _train(self, dataloader):
        assert self.torchnet.training == True
        for i, (img, gt, _) in enumerate(dataloader['labeled']):
            self.optim.zero_grad()
            img, gt = img.to(device), gt.to(device)
            pred_logit = self.torchnet(img)
            loss = self.criterion(pred_logit, gt.squeeze(1))
            loss.backward()
            self.optim.step()


class SemisupervisedTrainer(Trainer):
    alias = 'semi_train__'
    lrscheduler_keys = ['milestones', 'gamma']
    optim_keys = ['lr', 'weight_decay', 'amsgrad']
    flow_control_keys = ['update_labeled', 'update_unlabeled']
    criterion_keys = ['weight']

    @classmethod
    def set_flag(cls):
        flags.DEFINE_integer(cls.alias + 'max_epoch', default=200, help='max_epoch for semi training')
        flags.DEFINE_multi_integer(cls.alias + 'milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
                                   help='milestones for semi training')
        flags.DEFINE_float(cls.alias + 'gamma', default=0.8, help='gamma for lr_scheduler in semi training')
        flags.DEFINE_float(cls.alias + 'lr', default=0.001, help='lr for semi training')
        flags.DEFINE_float(cls.alias + 'weight_decay', default=0, help='weight_decay for semi training')
        flags.DEFINE_multi_float(cls.alias + 'weight', default=[1, 1], help='weight balance for CE for semi training')
        flags.DEFINE_string(cls.alias + 'loss_name', default='crossentropy', help='criterion used in the semi training')
        flags.DEFINE_string(cls.alias + 'optim_name', default='Adam', help='optimzer used in the semi training')
        flags.DEFINE_string(cls.alias + 'optim_option', default='{}', help='optimzer used in the semi training')
        flags.DEFINE_string(cls.alias + 'scheduler', default='MultiStepLR', help='scheduler used in the semi training')

    def __init__(self, torchnet, dataloader, hparam) -> None:
        super().__init__(torchnet)
        self.name = 'Semi_Supervised_Training'
        self.dataloader = dataloader
        self.hparam = copy.deepcopy(self._rm_alias(hparam))
        optim_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.optim_keys)
        optim_hparam.update(**eval(self.hparam['optim_option']))
        self.optim = getattr(torch.optim, self.hparam['optim_name'])(self.torchnet.parameters(), **optim_hparam)
        lrschduler_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.lrscheduler_keys)
        self.lrscheduler = getattr(torch.optim.lr_scheduler, self.hparam['scheduler'])(self.optim,
                                                                                       **lrschduler_hparam)
        self.mask_generation = getattr(utils.mask_gene, 'naiveway')
        criterion_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.criterion_keys)
        self.criterion = get_citerion(self.hparam['loss_name'], **criterion_hparam)
        self.criterion.to(device)

    def _train(self, dataloaders):
        assert self.torchnet.training == True
        for i, ((limg, lgt, _), (uimg, ugt, _)) in enumerate(zip(dataloaders['labeled'], dataloaders['unlabeled'])):
            if self.hparam['update_labeled']:
                self.optim.zero_grad()
                limg, lgt = limg.to(device), lgt.to(device)
                pred_logit = self.torchnet(limg)
                loss = self.criterion(pred_logit, lgt.squeeze(1))
                loss.backward()
                self.optim.step()
            if self.hparam['update_unlabeled']:
                self.optim.zero_grad()
                uimg, ugt = uimg.to(device), ugt.to(device)
                pred_logit = self.torchnet(uimg)
                mask = self.mask_generation(pred_logit)
                loss = self.criterion(pred_logit, mask)
                loss.backward()
                self.optim.step()


class TrainWrapper(ABC):

    def __init__(self, fullTrainer: FullysupervisedTrainer, semiTrainer: SemisupervisedTrainer, hparams) -> None:
        super().__init__()
        self.fulltrainer = fullTrainer
        self.semitrainer = semiTrainer
        if hparams['save_dir'] is not None:
            self.writername = 'runs/' + hparams['save_dir']
        else:
            self.writername = 'runs/' + self.writer_name
        self.writer = SummaryWriter(self.writername)
        self.fulltrainer.set_writer(self.writer)
        self.semitrainer.set_writer(self.writer)
        self.save_hparams(hparams, self.writername)
        self.logger = config_logger(logger, self.writername)

    def run_fully_training(self):
        self.fulltrainer.start_training(self.writername)

    def run_semi_training(self, hparam):
        if hparam['load_pretrain'] == True:
            try:
                self.logger.info('load checkpoint....')
                self.semitrainer.load_checkpoint(hparam['model_path'])
            except Exception as e:
                self.logger.error(e)
                print('recheck your --model_path')
                exit(1)

        self.semitrainer.start_training(self.writername)

    @property
    def writer_name(self):
        return self.generate_current_time() + '_' + self.generate_random_str()

    @staticmethod
    def generate_random_str(randomlength=16):
        """
        生成一个指定长度的随机字符串
        """
        import random
        random_str = ''
        base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
        length = len(base_str) - 1
        for i in range(randomlength):
            random_str += base_str[random.randint(0, length)]
        return random_str

    @staticmethod
    def generate_current_time():
        from time import strftime, localtime
        ctime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        return ctime

    @classmethod
    def save_hparams(cls, hparams, writername):
        hparams = copy.deepcopy(hparams)
        import pandas as pd
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(hparams.items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        file_name = os.path.join(writername, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        pd.Series(hparams).to_csv(os.path.join(writername, 'opt.csv'))

    def cleanup(self):
        self.writer.export_scalars_to_json(self.writername + '/json.json')
        self.writer.close()
