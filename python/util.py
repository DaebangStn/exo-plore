import os
import sys
import copy
import time
import json
from collections import deque
import socket
import shutil
import socket
import signal
import argparse
import subprocess
from os import path as osp
from datetime import datetime
from itertools import product
from collections import defaultdict
from random import seed as random_seed
from random import gauss, uniform, choice, randint
from math import sqrt, ceil, floor, isclose
from inspect import currentframe, getargvalues
from typing import Dict, Type, Tuple, List, Any, Optional, Union, Mapping
from typing_extensions import override

import pprint
import yaml
from git import Repo
import pandas as pd
import numpy as np
from numpy import linalg as LA
from pathlib import Path
from tqdm import tqdm
from glob import glob


PROJECT_ROOT = Path(osp.abspath(osp.join(osp.dirname(__file__), "..")))


def timestamp() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def current_git_hash(reduce: Optional[int] = None) -> str:
    repo = Repo(PROJECT_ROOT)
    commit_hash = str(repo.head.commit.hexsha)
    if reduce is not None:
        commit_hash = commit_hash[:reduce]
    return commit_hash

class IgnoreUndefinedLoader(yaml.SafeLoader):
    pass

def ignore_undefined_constructor(loader, node):
    # Ignore the unknown tag and return None or a default value
    return None

IgnoreUndefinedLoader.add_constructor(None, ignore_undefined_constructor)
