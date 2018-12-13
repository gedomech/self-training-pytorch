import numpy as np
import torch
import torchvision.utils as vutils
import shutil
import matplotlib.pyplot as plt
import csv
import warnings
import time
import torch.nn.functional as F
from torchnet.meter import AverageValueMeter
from torch.utils.data import DataLoader

from loss.loss import JensenShannonDivergence

warnings.filterwarnings('ignore')


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0), -1)
    tflat = target.view(input.size(0), -1)
    intersection = (iflat * tflat).sum(1)

    return float(((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())


def iou_loss(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            try:
                ious.append(float(intersection) / max(union, 1).cpu().data.numpy())
            except:
                ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def image_batch_generator(dataset, device):
    """
    This function generates batches containing (images, masks, paths)
    :param dataset: torch.utils.data.Dataset object to be loaded
    :param batch_size: size of the batch
    :param number_workers: number of threads used to load data
    :param device: torch.device object where images and masks will be located.
    :return: (images, masks, paths)
    """
    if not issubclass(type(dataset), DataLoader):
        raise TypeError("Input must be an instance of the torch.utils.data.Dataset class")

    try:
        _, data_batch = enumerate(dataset).__next__()
    except:
        labeled_loader_iter = enumerate(dataset)
        _, data_batch = labeled_loader_iter.__next__()
    img, mask, paths = data_batch
    return img.to(device), mask.to(device), paths


def save_models(nets_, nets_path_, score_meters=None, epoch=0, history_score_dict=None, ):
    """
    This function saves the parameters of the nets
    :param nets_: networks containing the parameters to be saved
    :param nets_path_: list of path where each net will be saved
    :param score_meters: list of torchnet.meter.AverageValueMeter objects corresponding with each net
    :param epoch: epoch which was obtained the scores
    :return:
    """
    history_score_dict['epoch'] = epoch

    for idx, net_i in enumerate(nets_):

        if (idx == 0) and (history_score_dict['enet'] < score_meters[idx].value()[0]):
            history_score_dict['enet'] = score_meters[idx].value()[0]
            # print('The highest dice score for ENet is {:.3f} in the test'.format(highest_dice_enet))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 1) and (history_score_dict['unet'] < score_meters[idx].value()[0]):
            history_score_dict['unet'] = score_meters[idx].value()[0]
            # print('The highest dice score for UNet is {:.3f} in the test'.format(highest_dice_unet))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 2) and (history_score_dict['segnet'] < score_meters[idx].value()[0]):
            history_score_dict['segnet'] = score_meters[idx].value()[0]
            # print('The highest dice score for SegNet is {:.3f} in the test'.format(highest_dice_segnet))
            torch.save(net_i.state_dict(), nets_path_[idx])

    return history_score_dict


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
        # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:
                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print('error in colorize.')
        return color_image


def showImages(board, image_batch, mask_batch, segment_batch):
    color_transform = Colorize()
    means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
    stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])
    # import ipdb
    # ipdb.set_trace()
    if image_batch.min() < 0:
        for i in range(3):
            image_batch[:, i, :, :] = (image_batch[:, i, :, :]) * stds[i] + means[i]

    board.image(image_batch[0], 'original image')
    board.image(color_transform(mask_batch[0]), 'ground truth image')
    board.image(color_transform(segment_batch[0]), 'prediction given by the net')


def learning_rate_decay(optims, factor=0.95):
    for param_group in optims.param_groups:
        param_group['lr'] = param_group['lr'] * factor


def learning_rate_reset(optims, lr=1e-4):
    for param_group in optims.param_groups:
        param_group['lr'] = lr


def map_(func, *list):
    return [*map(func, *list)]


def batch_labeled_loss_(img, mask, nets, criterion):
    loss_list = []
    prediction_list = []
    dice_score = []
    for net_i in nets:
        pred = net_i(img)
        labeled_loss = criterion(pred, mask.squeeze(1))
        loss_list.append(labeled_loss)
        ds = dice_loss(pred2segmentation(net_i(img)), mask.squeeze(1))
        dice_score.append(ds)
        prediction_list.append(pred)

    return prediction_list, loss_list, dice_score


def test(nets_, test_loader_, device, **kwargs):
    class_number = 2

    """
    This function performs the evaluation with the test set containing labeled images.
    """

    map_(lambda x: x.eval(), nets_)

    dice_meters_test = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    mv_dice_score_meter = AverageValueMeter()

    with torch.no_grad():
        for i, (img, mask, _) in enumerate(test_loader_):

            (img, mask) = img.to(device), mask.to(device)
            distributions = torch.zeros([img.shape[0], class_number, img.shape[2], img.shape[3]]).to(device)

            for idx, net_i in enumerate(nets_):
                pred_test = nets_[idx](img)

                distributions += F.softmax(pred_test, 1)
                dice_test = dice_loss(pred2segmentation(pred_test), mask.squeeze(1))
                dice_meters_test[idx].add(dice_test)

            distributions /= 3
            mv_dice_score = dice_loss(pred2segmentation(distributions), mask.squeeze(1))
            mv_dice_score_meter.add(mv_dice_score.item())

    map_(lambda x: x.train(), nets_)

    return [dice_meters_test[idx] for idx in range(3)], mv_dice_score_meter


def get_mv_based_labels(imgs, nets,strategy):
    assert strategy in ('hard', 'soft')
    class_number = 2
    prediction = []
    if strategy =='soft':

        distributions = torch.zeros([imgs.shape[0], class_number, imgs.shape[2], imgs.shape[3]]).to(imgs.device)
        for idx, (net_i) in enumerate(nets):
            pred = F.softmax(net_i(imgs),1)
            prediction.append(pred)
            distributions += pred
        distributions /= 3
        return pred2segmentation(distributions), prediction
    else:
        distributions = torch.zeros([imgs.shape[0], imgs.shape[2], imgs.shape[3]]).long().to(imgs.device)
        for idx, (net_i) in enumerate(nets):
            pred = F.softmax(net_i(imgs), 1)
            prediction.append(pred)
            distributions += pred.max(1)[1]
        distributions /= len(nets)
        distributions = (distributions<0.5).long()
        return distributions, prediction



def cotraining(prediction, pseudolabel, nets, criterion, device):
    loss = []
    for idx, net_i in enumerate(nets):
        unlabled_loss = criterion(prediction[idx], pseudolabel.to(device))
        loss.append(unlabled_loss)
    return loss


def get_loss(predictions):
    p = torch.cat(predictions)
    criteron = JensenShannonDivergence()
    loss = criteron(p)
    return loss


def visualize(writer, nets_, image_set, n_images, c_epoch, randomly=True, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Visualize n_images from the input set of images (image_set).
    :param nets_: networks used to extract the predictions from the input images
    :param image_set: set of images to be visualized
    :param n_images: number of images that really will be visualized
    :param c_epoch: current epoch
    :param randomly: indicates if n_images will be randomly taken from image_set

    The rest of parameters correspond to the input arguments of torchvision.utils.make_grid.
    For more documentation refers to https://pytorch.org/docs/stable/torchvision/utils.html
    :param nrow:
    :param padding:
    :param normalize:
    :param range:
    :param scale_each:
    :param pad_value:
    :return:
    """

    n_samples = np.min([image_set.shape[0], n_images])

    if randomly:
        idx = np.random.randint(low=0, high=image_set.shape[0], size=n_samples)
    else:
        idx = np.arange(n_samples)

    imgs = image_set[idx, :, :, :]
    for idx, net_i in enumerate(nets_):
        pred_grid = vutils.make_grid(net_i(imgs).cpu(), nrow=nrow, padding=padding, pad_value=pad_value,
                                     normalize=normalize, range=range, scale_each=scale_each)
        if idx == 0:
            writer.add_image('Enet Predictions', pred_grid, c_epoch)  # Tensor
        elif idx == 1:
            writer.add_image('Unet Predictions', pred_grid, c_epoch)  # Tensor
        else:
            writer.add_image('SegNet Predictions', pred_grid, c_epoch)  # Tensor


def s_forward_backward(net, optim, imgs, masks, criterion):
    now = time.time()
    optim.zero_grad()
    pred = net(imgs)
    loss = criterion(pred, masks.squeeze(1))
    loss.backward()
    optim.step()
    dice_score = dice_loss(pred2segmentation(pred), masks.squeeze(1))

    return dice_score


def evaluate(net, dataloader, device):
    net.eval()
    dice_meter = AverageValueMeter()
    dice_meter.reset()
    with torch.no_grad():
        for i, (img, mask, path) in enumerate(dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = net(img)
            pred_mask = pred2segmentation(pred)
            dice_meter.add(dice_loss(pred_mask, mask))

    net.train()
    return dice_meter.value()[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('enet_', 'best_model_'))


def plot_from_csvfile(csv_file: str, csv_file_baseline, delim=','):
    id_exp = csv_file[:-4].split('_')[-1]
    epoch, unlab, dev, lab, val = [], [], [], [], []
    with open(csv_file, 'r') as csvfile:
        data = csv.DictReader(csvfile, delimiter=delim)
        for row in data:
            epoch.append(int(row['epoch']))
            unlab.append(float(row['unlab']))
            dev.append(float(row['dev']))
            lab.append(float(row['lab']))
            val.append(float(row['val']))

    id_exp_baseline = csv_file_baseline[:-4].split('_')[-1]
    Epoch_baseline, unlab_baseline, dev_baseline, lab_baseline, val_baseline = [], [], [], [], []
    with open(csv_file_baseline, 'r') as csv_file_baseline:
        data = csv.DictReader(csv_file_baseline, delimiter=delim)
        for row in data:
            Epoch_baseline.append(int(row['epoch']))
            unlab_baseline.append(float(row['unlab']))
            dev_baseline.append(float(row['dev']))
            lab_baseline.append(float(row['lab']))
            val_baseline.append(float(row['val']))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(epoch, unlab, label='unlabaled')
    ax1.plot(epoch, dev, label='development')
    ax1.plot(epoch, lab, label='labeled')
    ax1.plot(epoch, val, label='validation')
    ax1.set_title('Pre-training_' + id_exp)
    # ax1.set_title('Net Performance on different subsets')
    ax2.plot(Epoch_baseline, unlab_baseline, label='unlabaled')
    ax2.plot(Epoch_baseline, dev_baseline, label='development')
    ax2.plot(Epoch_baseline, lab_baseline, label='labeled')
    ax2.plot(Epoch_baseline, val_baseline, label='validation')
    ax2.set_title('Semi-with pseudolabels   _' + id_exp_baseline)
    plt.legend()
    plt.show()

    if id_exp == id_exp_baseline:
        f.savefig('results_exp_for_{}_fraction.pdf'.format(id_exp), bbox_inches='tight')
    else:
        print('Warning: Files do not correspond to the same experiment!')


def save_segm2pdf(net, data_loader, batch_size, device, pdf_file, epoch):
    net.eval()
    list_names = data_loader.dataset.imgs[:batch_size]

    with torch.no_grad():
        imgs, mask, _ = image_batch_generator(data_loader, device=device)
        segms = pred2segmentation(net(imgs)).cpu().numpy()
        gts = mask.squeeze(dim=1).cpu().numpy()

        figs = plt.figure()
        f, axarr = plt.subplots(4, 2)
        for idx in range(len(list_names)):
            img_name = list_names[idx].split('/')[-1].split('.')[0]
            axarr[idx, 0].imshow(segms[idx], cmap='gray')
            axarr[idx, 0].set_title(img_name + ' Seg at epoch {}'.format(epoch), fontsize='x-small')
            axarr[idx, 0].axis('off')
            axarr[idx, 1].imshow(gts[idx], cmap='gray')
            axarr[idx, 1].set_title(img_name + ' GT at epoch {}'.format(epoch), fontsize='x-small')
            axarr[idx, 1].axis('off')

        pdf_file.savefig(f)

    net.train()


def extract_from_big_dict(big_dict, keys):
    """ Get a small dictionary with key in `keys` and value
        in big dict. If the key doesn't exist, give None.
        :param big_dict: A dict
        :param keys: A list of keys
    """
    #   TODO a bug has been found
    return {key: big_dict.get(key) for key in keys if big_dict.get(key, 'not_found') != 'not_found'}
