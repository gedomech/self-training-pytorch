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
    flags.DEFINE_string('model_path', default='', help='path to the pretrained model')
    flags.DEFINE_string('save_dir', default=None, help='path to save')
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


def load_checkpoint(torchnets, path: list):
    lab_dataloaders = []
    for path_i, net in zip(path, torchnets):
        model = torch.load(path_i, map_location=lambda storage, loc: storage)
        logger.info('Saved_epoch: {}, Dice: {:3f}'.format(model['epoch'], model['dice']))
        net.load_state_dict(model['model'])
        lab_dataloaders.append(model['labeled_data'])

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


def evaluate(epoch, nets, dataloader, name, writer, mode='eval', savedirs=None):
    with torch.no_grad():
        metrics = {}
        dices, dice_mv = _evaluate(dataloader['labeled'], nets, mode)
        logger.info('at epoch: {:3d}, under {} mode, labeled_data dice: {:.3f} '.format(epoch, mode, dice))
        metrics['%s/labeled' % name] = dice
        dices, dice_mv = _evaluate(dataloader['unlabeled'], nets, mode)
        logger.info('at epoch: {:3d}, under {} mode, unlabeled_data dice: {:.3f} '.format(epoch, mode, dice))
        metrics['%s/unlabeled' % name] = dice
        dices, dice_mv = _evaluate(dataloader['val'], nets, mode)
        logger.info('at epoch: {:3d}, under {} mode, val_data dice: {:.3f} '.format(epoch, mode, dice))
        metrics['%s/val' % name] = dice
    if mode == 'eval':
        writer.add_scalars(name, metrics, epoch)

        save_checkpoint(dice, epoch, savedirs)


def _evaluate(dataloader, nets, mode='eval'):
    assert mode in ('eval', 'train')
    dice_meter = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    dice_meter_mv = AverageValueMeter()
    if mode == 'eval':
        _ = [x.eval() for x in nets]
    else:
        _ = [x.train() for x in nets]

    with torch.no_grad():
        for i, (img, masks, _) in enumerate(dataloader):
            img, masks = img.to(device), masks.to(device)
            distributions = [], []
            for net in nets:
                pred_logit = net(img)
                distributions.append(pred_logit)
                pred_mask = pred2segmentation(pred_logit)
                dice_meter[i].add(dice_loss(pred_mask, masks))

        # compute pseudolabels based on the majority voting of ensemble models
        pseudolabels = compute_pseudolabels(distributions)
        dice_mv = compute_dice(pseudolabels.unsqueeze(1), masks)
        dice_meter_mv.add(dice_mv)

    if mode == 'eval':
        _ = [x.train() for x in nets]

    for net in nets:
        assert net.training == True, 'model is not set in train mode after evaluation stage'
    return [x.value()[0] for x in dice_meter], dice_meter_mv.value()[0]


def compute_dice(input, target):
    # with torch.no_grad:
    if input.shape[1] != 1: input = input.max(1)[1]
    smooth = 1.

    iflat = input.view(input.size(0),-1)
    tflat = target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)

    return float(((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())


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

    nets_path = [os.path.join(hparam['save_dir'], 'enet_0/semi_best.pth'),
                 os.path.join(hparam['save_dir'], 'enet_1/semi_best.pth'),
                 os.path.join(hparam['save_dir'], 'enet_2/semi_best.pth')]

    optimizers = [torch.optim.Adam(nets[0].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay']),
                  torch.optim.Adam(nets[1].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay']),
                  torch.optim.Adam(nets[2].parameters(), lr=hparam['lr'], weight_decay=hparam['lr_decay'])]

    schedulers = [MultiStepLR(optimizer=optimizers[0], milestones=hparam['milestones'], gamma=hparam['gamma']),
                  MultiStepLR(optimizer=optimizers[1], milestones=hparam['milestones'], gamma=hparam['gamma']),
                  MultiStepLR(optimizer=optimizers[2], milestones=hparam['milestones'], gamma=hparam['gamma'])]

    if hparam['save_dir'] is not None:
        writername = 'runs/' + hparam['save_dir']
    else:
        writername = 'runs/'
    writer = SummaryWriter(writername)

    # lab_data_iters = [iter(dataloader) for dataloader in labeled_loader_]
    criterion = get_unlabeled_loss(hparam['loss_name'])

    print("STARTING THE BASELINE TRAINING!!!!")
    for epoch in range(hparam['max_epoch']):
        evaluate(epoch + 1, mode='train', savedirs=nets_path)
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

    nets_path = [os.path.join(hparam['model_path'], 'enet_0/best.pth'),
                 os.path.join(hparam['model_path'], 'enet_1/best.pth'),
                 os.path.join(hparam['model_path'], 'enet_2/best.pth')]

    root = "datasets/ISIC2018"
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

    labeled_loader = load_checkpoint(nets, nets_path)
    unlabeled_loader = DataLoader(unlabeled_data, **unlabeled_loader_params)
    val_loader = DataLoader(val_data, **val_loader_params)

    data_loaders = {'labeled': labeled_loader,
                    'unlabeled': unlabeled_loader,
                    'val': val_loader}

    train_ensemble(nets, nets_path, data_loaders, hparam)
