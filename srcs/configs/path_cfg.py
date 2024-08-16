from typing import List, Optional, Union
import yacs.config

class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)
CN = Config

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# PATH CONFIG (currently set relative to the project root dir)
# -----------------------------------------------------------------------------

_C = CN()
_C.DATASET = CN()
_C.DATASET.DATA_LIST = CN()
_C.DATASET.TEXT = CN()

_C.ROOT_DIR = '..'
_C.DATASET.DATA_LIST.ROOT = '/home/ubuntu/xiajunkai/InteractTraj/srcs/utils/trainval'
_C.DATASET.DATA_PATH = '/home/ubuntu/DATA1/waymo'
_C.SAVE_DIR = 'results'

_C.LOGGER = 'wandb'# _C.LOGGER = 'tsboard'
