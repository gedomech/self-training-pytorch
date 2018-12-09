import os
import sys
import pandas as pd
import logging
import warnings
import torch

from absl import flags, app
from data.dataloader import ISICdata
from models.enet import Enet
from loss.loss import CrossEntropyLoss2d, JensenShannonDivergence
from utils.helpers import *
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

logger = logging.getLogger(__name__)
logger.parent = None
warnings.filterwarnings('ignore')


def get_default_parameter():
    flags.DEFINE_integer('num_workers', default=4, help='number of workers used in dataloader')
    flags.DEFINE_integer('batch_size', default=4, help='number of batch size')
    flags.DEFINE_boolean('semi_train__update_labeled', default=True,
                         help='update the labeled image while self training')
    flags.DEFINE_boolean('semi_train__update_unlabeled', default=True,
                         help='update the unlabeled image while self training')
    flags.DEFINE_boolean('run_pretrain', default=True,
                         help='run_pretrain')
    flags.DEFINE_boolean('run_semi', default=False,
                         help='run_self_training')
    flags.DEFINE_boolean('load_pretrain', default=True,
                         help='load_pretrain for self training')
    flags.DEFINE_string('model_path', default='checkpoints', help='path to the pretrained model')
    flags.DEFINE_string('loss_name', default='crossentropy', help='loss for semi supervised learning')
    flags.DEFINE_string('save_dir', default='semi', help='path to save')
    flags.DEFINE_float('labeled_percentate', default=1.0, help='how much percentage of labeled data you use')

    flags.DEFINE_integer('max_epoch', default=200, help='max_epoch for full training')
    flags.DEFINE_multi_integer('milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
                               help='milestones for full training')
    flags.DEFINE_float('gamma', default=0.5, help='gamma for lr_scheduler in full training')
    flags.DEFINE_float('lr', default=0.001, help='lr for full training')
    flags.DEFINE_float('lr_decay', default=0.2, help='decay of learning rate schedule')
    flags.DEFINE_multi_float('weight', default=[1, 1], help='weight balance for CE for full training')


def get_unlabeled_loss(lossname='crossentropy'):
    if lossname == 'crossentropy':
        criterion = CrossEntropyLoss2d([1, 1])
    elif lossname == 'jsd':
        criterion = JensenShannonDivergence(reduce=True, size_average=False)
    else:
        raise NotImplementedError
    return criterion


def load_checkpoint(labeled_data, torchnets, path: list):
    lab_dataloaders = []
    import copy
    for path_i, net in zip(path, torchnets):
        labeled_data_ = copy.deepcopy(labeled_data)
        model = torch.load(path_i, map_location=lambda storage, loc: storage)
        logger.info('Saved_epoch: {}, Dice: {:3f}'.format(model['epoch'], model['dice']))
        labeled_data_.dataset.imgs = model['labeled_data'].dataset.imgs
        labeled_data_.dataset.gts = model['labeled_data'].dataset.gts

        lab_dataloaders.append(labeled_data_)

    return lab_dataloaders


def batch_labeled_loss(img, mask, net, criterion):
    pred = net(img)
    labeled_loss = criterion(pred, mask.squeeze(1))
    ds = dice_loss(pred2segmentation(net(img)), mask.squeeze(1))
    return labeled_loss, ds, pred


def compute_pseudolabels(distributions: list):
    distributions = torch.cat([d.unsqueeze(0) for d in distributions], 0)
    return torch.mean(distributions, dim=0).max(1)[1]


def save_checkpoint(dice, nets, epoch, best_dice=-1, name=None):
    for net in nets:
        # save this checkpoint as last.pth
        dict2save = {}
        dict2save['epoch'] = epoch
        dict2save['dice'] = best_dice
        dict2save['model'] = net.save_dict
        if name is None:
            torch.save(dict2save, 'last.pth')
        else:
            torch.save(dict2save, name + '/last.pth')

        if dice > best_dice:
            best_dice = dice
            dict2save = dict()
            dict2save['epoch'] = epoch
            dict2save['dice'] = dice
            dict2save['model'] = net.state_dict()
            if name is None:
                torch.save(dict2save, 'best.pth')
            else:
                torch.save(dict2save, name + '/best.pth')
        else:
            return


def evaluate(epoch, nets, dataloader, name=None, writer=None, mode='eval', savedirs=None):
    with torch.no_grad():
        # dices  = _evaluate_mm(nets, dataloader['labeled'], mode)

        ## for the labeled data
        dice1 = _evaluate(net=nets[0], dataloader=dataloader['labeled'][0], mode='eval')
        dice2 = _evaluate(net=nets[1], dataloader=dataloader['labeled'][1], mode='eval')
        dice3 = _evaluate(net=nets[2], dataloader=dataloader['labeled'][2], mode='eval')

        print('labeled dataset:{},{},{}'.format(dice1, dice2, dice3))

        logger.info('at epoch: {:3d}, under {} mode, labeled_data dice: {:.3f} '.format(epoch, mode, dice))
        ## for unlabeled data

        dices = _evaluate_mm(nets, dataloader['unlabeled'], mode='eval')
        ## update data, for to log.
        print('unlabeled datset: {}, {}, {}'.format(dices[0], dices[1], dices[2]))

        ## for val data
        dices = _evaluate_mm(nets, dataloader['val'], mode='eval')

        print('val datset: {},{},{}'.format(dices[0], dices[1], dices[2]))
        # if mode == 'eval':
        #     writer.add_scalars(name, metrics, epoch)

        # save_checkpoint(dices, epoch, savedirs)


def _evaluate_mm(nets, dataloader, mode):
    return [_evaluate(net, dataloader, mode) for net in nets]


def _evaluate(net, dataloader, mode='eval'):
    assert mode in ('eval', 'train')
    dice_meter = AverageValueMeter()
    if mode == 'eval':
        net.eval()
    else:
        net.train()

    with torch.no_grad():
        for i, (img, gt, _) in enumerate(dataloader):
            img, gt = img.to(device), gt.to(device)
            pred_logit = net(img)
            pred_mask = pred2segmentation(pred_logit)
            dice_meter.add(dice_loss(pred_mask, gt))
    if mode == 'eval':
        net.train()
    assert net.training == True
    return dice_meter.value()[0]


def compute_dice(input, target):
    # with torch.no_grad:
    if input.shape[1] != 1: input = input.max(1)[1]
    smooth = 1.

    iflat = input.view(input.size(0), -1)
    tflat = target.view(input.size(0), -1)
    intersection = (iflat * tflat).sum(1)

    return float(((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())


def train_ensemble(nets_: list, data_loaders, hparam):
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    #  loading pre-trained models

    # _ = []
    #
    # map_(lambda x, y: [x.load_state_dict(torch.load(y, map_location='cpu')), x.train()], nets_, nets_path_)
    records = []
    historical_score_dict = {
        'epoch': -1,
        'enet_0': 0,
        'enet_1': 0,
        'enet_2': 0,
        'mv': 0,
        'jsd': 0}

    if not os.path.exists(hparam['save_dir']):
        os.mkdir(hparam['save_dir'])

    nets_path = [os.path.join(hparam['save_dir'], 'enet_0_semi_best.pth'),
                 os.path.join(hparam['save_dir'], 'enet_1_semi_best.pth'),
                 os.path.join(hparam['save_dir'], 'enet_2_semi_best.pth')]

    optimizers = [torch.optim.Adam(nets_[0].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay']),
                  torch.optim.Adam(nets_[1].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay']),
                  torch.optim.Adam(nets_[2].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay'])]

    schedulers = [MultiStepLR(optimizer=optimizers[0], milestones=hparam['milestones'], gamma=hparam['gamma']),
                  MultiStepLR(optimizer=optimizers[1], milestones=hparam['milestones'], gamma=hparam['gamma']),
                  MultiStepLR(optimizer=optimizers[2], milestones=hparam['milestones'], gamma=hparam['gamma'])]

    if hparam['save_dir'] is not None:
        writername = 'runs/' + hparam['save_dir']
    else:
        writername = 'runs/'
    writer = SummaryWriter(writername)

    criterion = get_unlabeled_loss(hparam['loss_name'])

    print("STARTING THE BASELINE TRAINING!!!!")
    for epoch in range(hparam['max_epoch']):
        evaluate(epoch + 1, nets=nets_, dataloader=data_loaders, mode='eval', writer=writer, savedirs=nets_path)
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, hparam['max_epoch']))

        # train with labeled data
        for _ in range(len(data_loaders['unlabeled'])):
            # train with labeled data
            llost_lst, prediction_lst, dice_score_lst = [], [], []
            for lab_loader, net_i in zip(data_loaders['labeled'], nets):
                imgs, masks, _ = image_batch_generator(lab_loader, device=device)
                prediction, llost, dice_score = batch_labeled_loss(imgs, masks, net_i, criterion)
                llost_lst.append(llost)
                prediction_lst.append(prediction)
                # dice_score_lst.append(dice_score)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(data_loaders['unlabeled'], device=device)
            pseudolabel, unlab_preds = get_mv_based_labels(imgs, nets_)
            uloss_lst = [0, 0, 0]
            if hparam['loss_name'] == 'crossentropy':
                criterion = criterion.to(device)
                uloss_lst = [criterion(unlab_preds, pseudolabel)
                             for unlab_preds, pseudolabel in zip(pseudolabel, unlab_preds)]
            elif hparam['loss_name'] == 'jsd':
                uloss_lst = [criterion(unlab_preds) for unlab_preds in pseudolabel]

            total_loss = [x + y for x, y in zip(llost_lst, uloss_lst)]
            for idx in range(len(optimizers)):
                optimizers[idx].zero_grad()
                total_loss[idx].backward()
                optimizers[idx].step()
                schedulers[idx].step()

        evaluate(epoch + 1, nets, data_loaders, 'eval', writer, mode='eval', savedirs=nets_path)

        # print(
        #     'train epoch {0:1d}/{1:d} ensemble: enet0_dice_score={2:.3f}, enet2_dice_score={3:.3f}'.format(
        #         epoch + 1, hparam['max_epoch'], dice_meters[0].value()[0], dice_meters[1].value()[0]))

        # evaluate(epoch, mode='eval', savedir=savedir)

        score_meters, ensemble_score = test(nets_, test_data, device=device)

        print(
            'val epoch {0:d}/{1:d} ensemble: enet0_dice={2:.3f}, enet2_dice={3:.3f}, with mv_dice={4:.3f}'.format(
                epoch + 1,
                hparam['max_epoch'],
                score_meters[0].value()[0],
                score_meters[1].value()[0],
                ensemble_score.value()[0]))

        historical_score_dict = save_models(nets_, nets_path, score_meters, epoch, historical_score_dict)
        if ensemble_score.value()[0] > historical_score_dict['mv']:
            historical_score_dict['mv'] = ensemble_score.value()[0]

        records.append(historical_score_dict)

        try:
            pd.DataFrame(records).to_csv('ensemble_records.csv')
        except Exception as e:
            print(e)


if __name__ == "__main__":
    get_default_parameter()
    hparam = flags.FLAGS.flag_values_dict()
    class_number = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## networks and optimisers
    nets = [Enet(class_number),
            Enet(class_number),
            Enet(class_number)]

    nets = map_(lambda x: x.to(device), nets)

    # class_weigth = torch.Tensor(hparam['weight'])
    # criterion = CrossEntropyLoss2d(class_weigth).to(device) if (
    #         torch.cuda.is_available()) else CrossEntropyLoss2d(class_weigth)
    # ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)

    nets_path = [os.path.join(hparam['model_path'], 'enet_0_best.pth'),
                 os.path.join(hparam['model_path'], 'enet_1_best.pth'),
                 os.path.join(hparam['model_path'], 'enet_2_best.pth')]

    root = "datasets/ISIC2018"
    labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                            dataAugment=False, equalize=False)
    unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                              dataAugment=False, equalize=False)
    val_data = ISICdata(root=root, model='val', mode='semi', transform=True,
                        dataAugment=False, equalize=False)

    unlabeled_loader_params = {'batch_size': hparam['num_workers'],
                               'shuffle': True,
                               'num_workers': hparam['batch_size'],
                               'pin_memory': True}
    val_loader_params = {'batch_size': hparam['num_workers'],
                         'shuffle': False,
                         'num_workers': hparam['batch_size'],
                         'pin_memory': True}

    labeled_loader = DataLoader(labeled_data, **unlabeled_loader_params)
    unlabeled_loader = DataLoader(unlabeled_data, **unlabeled_loader_params)
    val_loader = DataLoader(val_data, **val_loader_params)
    labeled_loader = load_checkpoint(labeled_loader, nets, nets_path)

    data_loaders = {'labeled': labeled_loader,
                    'unlabeled': unlabeled_loader,
                    'val': val_loader}

    # nets[0].load_state_dict(torch.load(nets_path[0], map_location=lambda storage, loc: storage)['model'])
    # nets[0].train()
    map_(lambda x, y: [x.load_state_dict(torch.load(y, map_location=lambda storage, loc: storage)['model']), x.train()],
         nets,
         nets_path)

    # dice = _evaluate(nets[0], dataloader=data_loaders['val'], mode='eval')
    # print(dice)
    # dice = _evaluate(nets[0], dataloader=data_loaders['labeled'][0], mode='train')
    # print(dice)
    # dice = _evaluate(nets[0], dataloader=data_loaders['unlabeled'], mode='eval')
    # print(dice)

    train_ensemble(nets, data_loaders, hparam)
