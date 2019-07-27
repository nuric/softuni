"""Unification MLP."""
import argparse
import json
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T

# Disable scientific printing
np.set_printoptions(suppress=True, precision=3, linewidth=180)
# pylint: disable=line-too-long

# Arguments
parser = argparse.ArgumentParser(description="Run UMLP on randomly generated tasks.")
# parser.add_argument("task", help="Task name to solve.")
parser.add_argument("name", help="Name prefix for saving files etc.")
parser.add_argument("-l", "--length", default=4, type=int, help="Fixed length of symbol sequences.")
parser.add_argument("-s", "--symbols", default=4, type=int, help="Number of symbols.")
parser.add_argument("-r", "--rules", default=3, type=int, help="Number of rules in repository.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-t", "--tsize", default=100, type=int, help="Per task training size.")
ARGS = parser.parse_args()

# ---------------------------

def rand_syms(symbols: int = None, length: int = None, replace: bool = False):
  """Return unique random symbols."""
  symbols = symbols or ARGS.symbols
  length = length or ARGS.length
  # We'll add 2, reserve 0 for padding, 1 for no answer,
  return np.random.choice(symbols, size=length, replace=replace) + 2

# Generate random data for tasks
def gen_task1() -> np.ndarray:
  """Task 1: head of random sequence."""
  seq = rand_syms()
  return np.concatenate((seq, [1], [seq[0]])) # (L+2,)

def gen_task2() -> np.ndarray:
  """Task 2: tail of random sequence."""
  seq = rand_syms()
  return np.concatenate((seq, [2], [seq[-1]])) # (L+2,)

def gen_task3() -> np.ndarray:
  """Task 3: item that is repeated twice."""
  seq = rand_syms()
  # Fail case
  if np.random.rand() < 0.5:
    return np.concatenate((seq, [3], [1]))
  # select two random locations and make them equal
  x, y = np.random.choice(len(seq), size=2, replace=False)
  seq[x] = seq[y] # item is repeated
  return np.concatenate((seq, [3], [seq[x]])) # (L+2,)

def gen_task4() -> np.ndarray:
  """Task 4: all items equal."""
  seq = rand_syms(replace=True)
  # Fail case
  if np.random.rand() < 0.5:
    while len(np.unique(seq)) == 1:
      seq = rand_syms(replace=True)
    return np.concatenate((seq, [4], [1])) # (L+2,)
  # all items equal
  seq[:] = seq[0]
  return np.concatenate((seq, [4], [seq[0]])) # (L+2,)

def gen_all(tsize: int = None) -> np.ndarray:
  """Generate all tasks."""
  data = list()
  for i in range(1, 5):
    f = globals()['gen_task'+str(i)]
    for _ in range(tsize or ARGS.tsize):
      data.append(f())
  return np.stack(data) # (tasks*S, L+2)

train_data = gen_all() # (S, L+2)
val_data = gen_all(50) # (T*50, L+2)
test_data = gen_all(50) # (T*50, L+2)

# ---------------------------
