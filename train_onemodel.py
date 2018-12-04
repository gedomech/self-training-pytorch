# coding=utf-8
import logging
import sys
import os

from absl import flags, app
from models.trainers import FullysupervisedTrainer, SemisupervisedTrainer, TrainWrapper
from data.dataloader import get_dataloader, get_exclusive_dataloaders
from models.enet import Enet
from utils.helpers import *

logger = logging.getLogger(__name__)
logger.parent = None
sys.path.extend([os.path.dirname(os.getcwd())])
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    flags.DEFINE_integer('idx_model', default=0, help='indicate the index (0, 1 or 2) of the model to be pre-trained')


def run(argv):
    del argv

    hparam = flags.FLAGS.flag_values_dict()

    # data for semi-supervised training
    # data_loaders = get_dataloader(hparam)
    data_loaders = get_exclusive_dataloaders(hparam)

    # selecting the labeled dataset to be used for the model
    data_loaders['labeled'] = data_loaders['labeled'][hparam['idx_model']]

    # networks and optimisers
    net = Enet(2)
    net = net.to(device)

    fully_trainer = FullysupervisedTrainer(net, data_loaders, hparam)
    semi_trainer = SemisupervisedTrainer(net, data_loaders, hparam)
    train_wrapper = TrainWrapper(fully_trainer, semi_trainer, hparam)
    if hparam['run_pretrain']:
        train_wrapper.run_fully_training()

    if hparam['run_semi']:
        train_wrapper.run_semi_training(hparam)
    train_wrapper.cleanup()


if __name__ == '__main__':
    FullysupervisedTrainer.set_flag()
    SemisupervisedTrainer.set_flag()
    get_default_parameter()
    app.run(run)
