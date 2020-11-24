import random

import numpy as np

from python.util import *

import ray
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

import dill
from dill import Unpickler
from io import BytesIO
from python.dummy import Dummy
import pickle
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

@ray.remote
class SeedManager:
    def __init__(self, seed):
        self.seed = seed

    def get_seed(self):
        self.seed += 1
        return self.seed


@ray.remote(num_gpus=1)
class CudaChecker:
    def __init__(self):
        self._hostname = socket.gethostname()
        self._cuda_available = torch.cuda.is_available()

    def check(self) -> Tuple[bool, str]:
        return self._cuda_available, self._hostname

def seed_all(seed: int):
    pass
    # random_seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # local_mode = ray.worker._mode() == ray.worker.LOCAL_MODE
    # return device if not local_mode else "cpu"
    return device

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, default="small_pc4")
    parser.add_argument("-f", "--config-file", type=str, default="python/config.py")
    parser.add_argument("-m", "--metadata", type=str, default="data/gait_generator/debug.yaml")
    parser.add_argument("-c", "--checkpoint", type=str, help="Specific checkpoint path")
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-e', '--epoch', type=int, default=None)
    parser.add_argument('-s', '--seed', type=int, default=randint(0, 1000))
    return parser.parse_args()


def is_float(value: str) -> bool:
    """Helper function to check if a string represents a float."""
    if value.count('.') == 1:  # Check if there's exactly one decimal point
        # Split the string at the decimal point and check if both parts are numeric
        left, right = value.split('.')
        if left.isdigit() and right.isdigit():
            return True
        elif left == '' and right.isdigit():  # Handling cases like ".123"
            return True
    return False


def parse_metadata(metadata: str) -> Dict[str, Any]:
    """ Parse metadata string and return a dictionary.
    :param metadata:
        1. In many cases, first string is its key
        2. The comment starts with '#' and is ignored
    :return:
    """
    results = {}
    for line in metadata.strip().splitlines():
        if not line.strip() or line.strip().startswith('#'):
            continue
        # delete comments and split by space
        parts = line.split('#')[0].strip().split()
        if len(parts) < 2:
            continue
        key = parts[0]
        value = parts[1]
        if value.isdigit():
            value = int(value)
        elif is_float(value):
            value = float(value)
        elif value in ['True', 'False', 'true', 'false']:
            value = value.lower() == 'true'
        results[key] = value
    return results


def debug_compute_graph(model: torch.nn.Module, y: torch.Tensor, x: Optional[torch.Tensor] = None):
    from torchviz import make_dot
    if x:
        params = {"x": x, **dict(model.named_parameters())}
    else:
        params = dict(model.named_parameters())
    graph = make_dot(y, params=params)
    graph.view()


def append_comment_to_metadata(metadata: str, comment: Union[str, List[str]]) -> str:
    if isinstance(comment, str):
        comment = [comment]
    comment = [f'# {c}' for c in comment]
    return '\n'.join(comment) + '\n' + metadata


def mean_dict_list(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums, counts = defaultdict(float), defaultdict(int)
    for d in data:
        for key, value in d.items():
            sums[key] += value
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums.keys()}
