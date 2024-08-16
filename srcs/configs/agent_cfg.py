from typing import List, Optional, Union
import yacs.config
from .path_cfg import _C as _C_PATH

class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
# task config can be a list of conifgs like "A.yaml,B.yaml"
_C.SEED = 0
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "default"

_C.EXPERIMENT_DIR = "results/debug"
_C.EXPERIMENT_NAME = "pipeline"
_C.TENSORBOARD_DIR = "tensorboard"
_C.WANDB_PROJ = "test"
_C.GPU = None # GPU id (-1 if use CPU)
_C.SAVE_CHECKPOINT = False

# Number of model updates during training
# _C.model = "meta/llama-3.1-8b-instruct"
# _C.api_key = "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"

_C.model = "meta/llama-3.1-70b-instruct"
_C.api_key = "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"

# _C.model = "meta/llama-3.1-405b-instruct"
# _C.api_key = "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"



_C.base_url = "https://integrate.api.nvidia.com/v1"

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    # config.FINETUNE = finetune_config.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)
    config.freeze()
    return config
