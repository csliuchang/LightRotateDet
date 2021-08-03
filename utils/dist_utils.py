from torch import distributed as dist
from getpass import getuser
from socket import gethostname
from distutils.version import LooseVersion
import torch

TORCH_VERSION = torch.__version__


def get_dist_info():
    if LooseVersion(TORCH_VERSION) < LooseVersion('1.0'):
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_host_info():
    return f'{getuser()}@{gethostname()}'