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
parser.add_argument("-i", "--invariants", default=3, type=int, help="Number of invariants per task.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-t", "--tsize", default=100, type=int, help="Random data generations per task.")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Training batch size.")
ARGS = parser.parse_args()

LENGTH = ARGS.length
EMBED = ARGS.embed
# We'll add 2, reserve 0 for padding, 1 for no answer,
VOCAB = ARGS.symbols + 2
FOLDS = 5

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
  return np.concatenate(([1], seq, [seq[0]])) # (1+L+1,)

def gen_task2() -> np.ndarray:
  """Task 2: tail of random sequence."""
  seq = rand_syms()
  return np.concatenate(([2], seq, [seq[-1]])) # (1+L+1,)

def gen_task3() -> np.ndarray:
  """Task 3: item that is repeated twice."""
  seq = rand_syms()
  # Fail case
  if np.random.rand() < 0.5:
    return np.concatenate(([3], seq, [1]))
  # select two random locations and make them equal
  x, y = np.random.choice(len(seq), size=2, replace=False)
  seq[x] = seq[y] # item is repeated
  return np.concatenate(([3], seq, [seq[x]])) # (1+L+1,)

def gen_task4() -> np.ndarray:
  """Task 4: all items equal."""
  seq = rand_syms(replace=True)
  # Fail case
  if np.random.rand() < 0.5:
    while len(np.unique(seq)) == 1:
      seq = rand_syms(replace=True)
    return np.concatenate(([4], seq, [1])) # (1+L+1,)
  # all items equal
  seq[:] = seq[0]
  return np.concatenate(([4], seq, [seq[0]])) # (1+L+1,)

def gen_all(tsize: int = None, unique: bool = True) -> np.ndarray:
  """Generate all tasks."""
  data = list()
  for i in range(1, 5):
    f = globals()['gen_task'+str(i)]
    for _ in range(tsize or ARGS.tsize):
      data.append(f())
  return np.unique(np.stack(data), axis=0) # (tasks*S, 1+L+1)

data = gen_all() # (S, 1+L+1)
nfolds = C.datasets.get_cross_validation_datasets_random(data, FOLDS) # 5 folds, list of 5 tuples train/test

print("Data:", data.shape)
print("Tasks:", np.unique(data[:,0], return_counts=True))
print("# Folds:", len(nfolds))
print("# Train:", len(nfolds[0][0]))
print("# Test:", len(nfolds[0][1]))

# ---------------------------

# Unification Network
class UMLP(C.Chain):
  """Unification feed-forward network."""
  def __init__(self, inv_examples):
    super().__init__()
    self.add_persistent('inv_examples', inv_examples) # (T, I, 1+L+1)
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(VOCAB, EMBED, ignore_label=0)
      self.vmap_params = C.Parameter(0.0, (inv_examples.shape[:2]) + (VOCAB,), name='vmap_params')
      self.l1 = L.Linear(LENGTH*EMBED, EMBED*2)
      self.l2 = L.Linear(EMBED*2, EMBED)
      self.l3 = L.Linear(EMBED, VOCAB)

  def predict(self, example):
    """Just a forward prediction of given example."""
    # example (B, ..., L)
    ex = self.embed(example) # (B, ..., L, E)
    flat_ex = F.reshape(ex.shape[:-2] + (-1,)) # (B, ..., L*E)
    nbaxes = len(example.shape)-1
    out = F.tanh(self.l1(flat_ex, n_batch_axes=nbaxes)) # (B, ..., E*2)
    out = F.tanh(self.l2(out, n_batch_axes=nbaxes)) # (B, ..., E)
    return self.l3(out, n_batch_axes=nbaxes) # (B, ..., V)

  def forward(self, ground_examples):
    """Compute the forward inference pass for given stories."""
    # ground_examples (B, 1+L+1)
    pass

# Wrapper chain for training
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    self.add_persistent('uniparam', 0.0)
    with self.init_scope():
      self.predictor = predictor

  def forward(self, ground_examples):
    """Compute total loss to train."""
    # ground_examples (B, 1+L+1)
    import ipdb; ipdb.set_trace()
    print("HERE")
    pass

# ---------------------------

# Training tools
def select_invariants(data: list, taskid: int):
  """Select I many examples with different answers."""
  invariants = list()
  answers = set()
  # This is shuffled initially
  for d in data:
    if d[0] == taskid and d[-1] not in answers and len(invariants) < ARGS.invariants:
      invariants.append(d)
      answers.add(d[-1])
    if len(invariants) == ARGS.invariants:
      break
  if len(invariants) < ARGS.invariants:
    raise ValueError("Not enough symbols to generate multiple invariants.")
  return invariants

def enable_unification(trainer):
  """Enable unification loss function in model."""
  trainer.updater.get_optimizer('main').target.uniparam = 1.0

# ---------------------------

# Training on single fold
def train(train_data, test_data, idx: int = 0):
  """Train new UMLP on given data."""
  # Setup invariant repositories
  # we'll take I many examples for each task with different answers for each fold
  invariants = [np.stack(select_invariants(train_data, i)) for i in range(1, 5)] # T x (I, 1+L+1)
  invariants = np.stack(invariants) # (T, I, 1+L+1)
  # ---------------------------
  # Setup model
  model = UMLP(invariants)
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam().setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  updater = T.StandardUpdater(train_iter, optimiser, device=-1)
  trainer = T.Trainer(updater, (100, 'epoch'))
  # ---------------------------
  # Setup trainer extensions
  trainer.extend(enable_unification, trigger=(40, 'epoch'))
  test_iter = C.iterators.SerialIterator(test_data, 128, repeat=False, shuffle=False)
  trainer.extend(T.extensions.Evaluator(test_iter, cmodel, device=-1), name='test')
  trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
  trainer.extend(T.extensions.LogReport(log_name=ARGS.name+'_log.json'))
  trainer.extend(T.extensions.FailOnNonNumber())
  # report_keys = ['loss', 'vmap', 'uatt', 'oatt', 'batt', 'rpred', 'opred', 'uni', 'oacc', 'acc']
  # trainer.extend(T.extensions.PrintReport(['epoch'] + ['main/'+s for s in report_keys] + [p+'/main/'+s for p in ('val', 'test') for s in ('loss', 'acc')] + ['elapsed_time']))
  # ---------------------------
  trainer.run()

# ---------------------------

# Training loop
try:
  for i, (traind, testd) in enumerate(nfolds):
    train(traind, testd, i)
except KeyboardInterrupt:
  if ARGS.debug:
    import ipdb; ipdb.set_trace()
