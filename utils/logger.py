# coding=utf-8
import logging
import sys
import os
from absl import flags

logger = logging.getLogger(__name__)
logger.parent = None


def config_logger(log_dir):
    """ Get console handler """
    log_format = logging.Formatter("[%(module)s - %(asctime)s - %(levelname)s] %(message)s")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_format)

    fh = logging.FileHandler(os.path.join(log_dir, 'log.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(log_format)

    logger.handlers = [console_handler, fh]


def get_default_parameter():
    flags.DEFINE_integer('num_workers', default=4, help='number of workers used in dataloader')
    flags.DEFINE_integer('batch_size', default=4, help='number of batch size')
    flags.DEFINE_boolean('semi_train__update_labeled', default=True,
                         help='update the labeled image while self training')
    flags.DEFINE_boolean('semi_train__update_unlabeled', default=True,
                         help='update the unlabeled image while self training')
    flags.DEFINE_boolean('run_pretrain', default=False,
                         help='run_pretrain')
    flags.DEFINE_boolean('run_semi', default=False,
                         help='run_self_training')
    flags.DEFINE_boolean('load_pretrain', default=True,
                         help='load_pretrain for self training')
    flags.DEFINE_string('model_path', default='', help='path to the pretrained model')
    flags.DEFINE_string('save_dir', default=None, help='path to save')
    flags.DEFINE_float('labeled_percentate', default=1.0, help='how much percentage of labeled data you use')
