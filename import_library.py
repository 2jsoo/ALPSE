import math
import logging
from functools import partial
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
import neurokit2 as nk
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR


from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.optim import create_optimizer

from types import SimpleNamespace

import gc

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, KFold
import random
import copy


import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from collections import Counter
import gc


import multiprocessing
from multiprocessing import Process, Manager
from contextlib import contextmanager


import wfdb
from wfdb import processing
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import neurokit2 as nk
from collections import Counter
import pickle
import pandas as pd
import torch

import ray

import pywt