import io
import os
import socket

import torch as th
import torch.distributed as dist


def setup_dist(devices=''):
  if devices != '':
      os.environ["CUDA_VISIBLE_DEVICES"] = devices.split(',')[0]


def dev():
  if th.cuda.is_available():
      return th.device("cuda")
  if hasattr(th.backends, "mps") and th.backends.mps.is_available():
      return th.device("mps")
  return th.device("cpu")


def load_state_dict(path, **kwargs):
  with open(path, "rb") as f:
      data = f.read()
  return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
  pass