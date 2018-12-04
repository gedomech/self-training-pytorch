# coding=utf-8
import logging
import sys
import os

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



