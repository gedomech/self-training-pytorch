# coding=utf-8
import logging
import sys
import os

from absl import flags, app
from models.trainers import FullysupervisedTrainer, SemisupervisedTrainer, TrainWrapper
from data.dataloader import get_dataloader
from models.enet import Enet
from utils.helpers import *
from utils.logger import get_default_parameter

logger = logging.getLogger(__name__)
logger.parent = None
sys.path.extend([os.path.dirname(os.getcwd())])
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(argv):
    del argv

    hparam = flags.FLAGS.flag_values_dict()

    # data for semi-supervised training
    data_loaders = get_dataloader(hparam)

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
