# coding=utf-8
import os
from argparse import Namespace, ArgumentParser
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def mPool(func, **args):
    return list(Pool().starmap(func, list(zip(*args.values()))))


def isAllowedExtension(path: Path) -> bool:
    return True if path.suffix.lower() in ('.jpg', '.png') else False


def isFolderHasAllowedFiles(path: Path) -> bool:
    namelist = [x for x in os.listdir(path) if isAllowedExtension(Path(x))]
    return True if namelist.__len__() > 0 else False


def slides_copy(path_in: Path, resolution: tuple, path_out: Path) -> None:
    img = Image.open(path_in)
    img = img.resize(size=resolution)
    assert img.size == resolution
    img.save(path_out)


def main(args: Namespace) -> None:
    subfolder_path = [Path(x) for x in os.listdir(args.datapath) if
                      os.path.isdir(x) and isFolderHasAllowedFiles(Path(x))]

    for folder in subfolder_path:
        filenames = [Path(x) for x in os.listdir(folder) if isAllowedExtension(Path(x))]
        new_folder = Path(folder.name + args.suffix)
        if not new_folder.exists():
            new_folder.mkdir()
        mPool(slides_copy, path_in=[folder.joinpath(x) for x in filenames], resolution=repeat(args.resolution),
              path_out=[new_folder.joinpath(x) for x in filenames])


def argument_parse():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='./')
    parser.add_argument('--resolution', default=(256, 256))
    parser.add_argument('--suffix', default='_temp')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parse()
    main(args)
