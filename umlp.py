"""Unification MLP."""
import argparse
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)
# pylint: disable=line-too-long

# Arguments
parser = argparse.ArgumentParser(description="Run UMLP on randomly generated tasks.")
# parser.add_argument("task", help="Task name to solve.")
parser.add_argument("name", help="Name prefix for saving files etc.")
parser.add_argument("-l", "--length", default=4, type=int, help="Fixed length of symbol sequences.")
parser.add_argument("-s", "--symbols", default=8, type=int, help="Number of symbols.")
parser.add_argument("-i", "--invariants", default=1, type=int, help="Number of invariants per task.")
parser.add_argument("-e", "--embed", default=16, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-nu", "--nouni", action="store_true", help="Disable unification.")
parser.add_argument("-t", "--tsize", default=0, type=int, help="Training size per task, 0 to use everything.")
parser.add_argument("-g", "--gsize", default=1000, type=int, help="Random data tries per task.")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Training batch size.")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("-o", "--outf", default="{name}_l{length}_s{symbols}_i{invariants}_e{embed}_t{tsize}_f{foldid}")
ARGS = parser.parse_args()

LENGTH = ARGS.length
EMBED = ARGS.embed
# We'll add 1, reserve 0 for padding
VOCAB = ARGS.symbols + 1
FOLDS = 5

# ---------------------------

def rand_syms(symbols: int = None, length: int = None, replace: bool = True):
  """Return unique random symbols."""
  symbols = symbols or ARGS.symbols
  length = length or ARGS.length
  # We'll add 1, reserve 0 for padding
  return np.random.choice(symbols, size=length, replace=replace) + 1

# Generate random data for tasks
def gen_task1() -> np.ndarray:
  """Task 1: return constant symbol."""
  seq = rand_syms()
  return np.concatenate(([1], seq, [2])) # (1+L+1,)

def gen_task2() -> np.ndarray:
  """Task 2: head of random sequence."""
  seq = rand_syms()
  return np.concatenate(([2], seq, [seq[0]])) # (1+L+1,)

def gen_task3() -> np.ndarray:
  """Task 3: tail of random sequence."""
  seq = rand_syms()
  return np.concatenate(([3], seq, [seq[-1]])) # (1+L+1,)

def gen_task4() -> np.ndarray:
  """Task 4: item that is repeated twice."""
  seq = rand_syms(replace=False)
  # select two random locations and make them equal
  x, y = np.random.choice(len(seq), size=2, replace=False)
  seq[x] = seq[y] # item is repeated
  return np.concatenate(([4], seq, [seq[x]])) # (1+L+1,)

TASKS = 4

def gen_all() -> np.ndarray:
  """Generate all tasks."""
  gdata = list()
  for i in range(1, TASKS+1):
    f = globals()['gen_task'+str(i)]
    for _ in range(ARGS.gsize):
      gdata.append(f())
  gdata = np.unique(np.stack(gdata), axis=0) # (tasks*S, 1+L+1)
  np.random.shuffle(gdata)
  return gdata

data = gen_all() # (S, 1+L+1)
nfolds = C.datasets.get_cross_validation_datasets_random(data, FOLDS) # 5 folds, list of 5 tuples train/test

metadata = {'data': data.shape, 'tasks': np.unique(data[:, 0], return_counts=True),
            'folds': len(nfolds), 'train': len(nfolds[0][0]), 'test': len(nfolds[0][1])}
print(metadata)

# ---------------------------

def seq_rnn_embed(exs, birnn, init_state=None, return_sequences: bool = False):
  """Embed given sequences using rnn."""
  # exs.shape == (..., S, E)
  seqs = F.reshape(exs, (-1,)+exs.shape[-2:]) # (X, S, E)
  toembed = F.separate(seqs, 0) # X x [(S1, E), (S2, E), ...]
  hs, ys = birnn(init_state, toembed) # (2, X, E), X x [(S1, 2*E), (S2, 2*E), ...]
  if return_sequences:
    ys = F.stack(ys) # (X, S, 2*E)
    ys = F.reshape(ys, exs.shape[:-1] + (-1,)) # (..., S, 2*E)
    return ys
  hs = F.moveaxis(hs, 0, -2) # (X, 2, E)
  hs = F.reshape(hs, exs.shape[:-2] + (-1,)) # (..., 2*E)
  return hs

# Unification Network
class UMLP(C.Chain):
  """Unification feed-forward network."""
  def __init__(self, inv_examples):
    super().__init__()
    self.add_persistent('inv_examples', inv_examples) # (T, I, 1+L+1)
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(VOCAB, EMBED, ignore_label=0)
      self.task_embed = L.EmbedID(TASKS, EMBED)
      self.vmap_params = C.Parameter(0.0, (inv_examples.shape[:2]) + (VOCAB,), name='vmap_params')
      self.uni_birnn = L.NStepBiGRU(1, EMBED, EMBED, 0)
      self.uni_linear = L.Linear(EMBED*2, EMBED, nobias=True)
      self.l1 = L.Linear(LENGTH*EMBED+TASKS, EMBED*2)
      self.l2 = L.Linear(EMBED*2, EMBED)
      self.l3 = L.Linear(EMBED, EMBED)
    self.log = None

  def tolog(self, key, value):
    """Append to log dictionary given key value pair."""
    loglist = self.log.setdefault(key, [])
    loglist.append(value)

  def predict(self, combined_x):
    """Forward pass for combined input."""
    # combined_x (..., L*E+T)
    nbaxes = len(combined_x.shape)-1
    out = F.tanh(self.l1(combined_x, n_batch_axes=nbaxes)) # (..., E*2)
    out = F.tanh(self.l2(out, n_batch_axes=nbaxes)) # (..., E)
    out = self.l3(out, n_batch_axes=nbaxes) # (..., E)
    out = out @ self.embed.W.T # (..., V)
    return out

  def embed_predict(self, examples):
    """Just a forward prediction of given example."""
    # examples (..., 1+L)
    ex = self.embed(examples[..., 1:]) # (..., L, E)
    task_id = F.embed_id(examples[..., 0]-1, np.eye(TASKS, dtype=np.float32)) # (..., T)
    flat_ex = F.reshape(ex, ex.shape[:-2] + (-1,)) # (..., L*E)
    combined_ex = F.concat((flat_ex, task_id), axis=-1) # (..., L*E+T)
    return self.predict(combined_ex) # (..., V)

  def compute_ground_loss(self, examples, log_prefix=''):
    """Compute loss and accuracy on ground examples."""
    # examples (..., 1+L+1)
    preds = self.embed_predict(examples[..., :-1]) # (..., V)
    preds = F.reshape(preds, (-1, VOCAB)) # (..., V)
    targets = F.flatten(examples[..., -1]) # (...,)
    loss = F.softmax_cross_entropy(preds, targets) # ()
    acc = F.accuracy(preds, targets) # ()
    self.tolog(log_prefix+'loss', loss)
    self.tolog(log_prefix+'acc', acc)
    return preds # (..., V)

  def forward(self, ground_examples):
    """Compute the forward inference pass for given stories."""
    # ground_examples (B, 1+L+1)
    self.log = dict()
    # ---------------------------
    # Invariant ground prediction
    self.compute_ground_loss(self.inv_examples, log_prefix='ig')
    # Ground example prediction
    self.compute_ground_loss(ground_examples, log_prefix='o')
    # ---------------------------
    # Unification case
    task_ids = ground_examples[:, 0] # (B,)
    ground_inputs = ground_examples[:, 1:-1] # (B, L)

    invariant_inputs = self.inv_examples[..., 1:-1] # (T, I, L)
    invs_inputs = invariant_inputs[task_ids-1] # (B, I, L)

    # Compute variable map
    vmap = F.sigmoid(self.vmap_params*10) # (T, I, V)
    self.tolog('vmap', vmap)
    vmap = vmap[task_ids-1] # (B, I, V)
    vmap = vmap[np.arange(vmap.shape[0])[:, None, None], np.arange(vmap.shape[1])[None, :, None], invs_inputs] # (B, I, L)

    # Embed ground examples
    eg = self.embed(ground_inputs) # (B, L, E)
    ei = self.embed(invs_inputs) # (B, I, L, E)

    # Embed tasks for RNN init states
    embed_tasks = self.task_embed(task_ids-1) # (B, E)
    embed_tasks = F.repeat(embed_tasks[None, ...], 2, axis=0) # (2, B, E)

    # Extract unification features
    ground_rnn = seq_rnn_embed(eg, self.uni_birnn, init_state=embed_tasks, return_sequences=True) # (B, L, 2*E)
    embed_tasks = F.repeat(embed_tasks, invs_inputs.shape[1], axis=1) # (2, B*I, E)
    invs_rnn = seq_rnn_embed(ei, self.uni_birnn, init_state=embed_tasks, return_sequences=True) # (B, I, L, 2*E)
    ground_rnn = self.uni_linear(ground_rnn, n_batch_axes=2) # (B, L, E)
    invs_rnn = self.uni_linear(invs_rnn, n_batch_axes=3) # (B, I, L, E)
    # (B, I, L, E) x (B, L, E) -> (B, I, L, L)
    uni_att = F.einsum("ijke,ile->ijkl", invs_rnn, ground_rnn) # (B, I, L, L)
    uni_att = F.softmax(uni_att, axis=-1) # (B, I, L, L)
    self.tolog('uniatt', uni_att)

    # (B, I, L, L) x (B, L, E) -> (B, I, L, E)
    eu = F.einsum("ijkl,ile->ijke", uni_att, eg) # (B, I, L, E)

    # uni_embed = vmap[..., None]*eg[:, None] + (1-vmap)[..., None]*ei # (B, I, L, E)
    uni_embed = vmap[..., None]*eu + (1-vmap)[..., None]*ei # (B, I, L, E)
    uni_embed = F.reshape(uni_embed, uni_embed.shape[:-2] + (-1,)) # (B, I, L*E)

    # Make the prediction on the unification
    ets = F.embed_id(task_ids-1, np.eye(TASKS, dtype=np.float32)) # (B, T)
    ets = F.repeat(ets[:, None], vmap.shape[1], axis=1) # (B, I, T)
    uni_inputs = F.concat((uni_embed, ets), axis=-1) # (B, I, L*E+T)
    uni_preds = self.predict(uni_inputs) # (B, I, V)

    # Aggregate results from each invariant
    final_uni_preds = F.sum(uni_preds, -2) # (B, V)
    # ---------------------------
    return final_uni_preds # (B, V)

# Wrapper chain for training
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    self.add_persistent('uniparam', not ARGS.nouni)
    with self.init_scope():
      self.predictor = predictor

  def forward(self, ground_examples):
    """Compute total loss to train."""
    # ground_examples (B, 1+L+1)
    report = dict()
    # Compute main loss
    predictions = self.predictor(ground_examples) # (B, V)
    targets = ground_examples[:, -1] # (B,)
    uloss = F.softmax_cross_entropy(predictions, targets) # ()
    uacc = F.accuracy(predictions, targets) # ()
    report['uloss'] = uloss
    report['uacc'] = uacc
    # ---------------------------
    # Aux lossess
    for k in ['ig', 'o']:
      report[k+'loss'] = self.predictor.log[k+'loss'][0]
      report[k+'acc'] = self.predictor.log[k+'acc'][0]

    vloss = F.sum(self.predictor.vmap_params) # ()
    report['vloss'] = vloss
    # ---------------------------
    C.report(report, self)
    return self.uniparam*(uloss + 0.1*vloss + report['igloss']) + (1-self.uniparam)*report['oloss']

# ---------------------------

# Training tools
def select_invariants(data: list, taskid: int):
  """Select I many examples with different answers."""
  data = np.stack(data) # (S, 1+L+1)
  np.random.shuffle(data)
  return data[data[:, 0] == taskid][:ARGS.invariants]

def print_vmap(trainer):
  """Enable unification loss function in model."""
  print(trainer.updater.get_optimizer('main').target.predictor.inv_examples)
  print(F.sigmoid(trainer.updater.get_optimizer('main').target.predictor.vmap_params*10))

# ---------------------------

# Training on single fold
def train(train_data, test_data, foldid: int = 0):
  """Train new UMLP on given data."""
  # Setup invariant repositories
  # we'll take I many examples for each task with different answers for each fold
  invariants = [select_invariants(train_data, i) for i in range(1, 5)] # T x (I, 1+L+1)
  invariants = np.stack(invariants) # (T, I, 1+L+1)
  # ---------------------------
  # Setup model
  model = UMLP(invariants)
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam(alpha=ARGS.learning_rate).setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  updater = T.StandardUpdater(train_iter, optimiser, device=-1)
  trainer = T.Trainer(updater, (2000, 'iteration'), out='results/umlp_result')
  # ---------------------------
  fname = ARGS.outf.format(**vars(ARGS), foldid=foldid)
  # Setup trainer extensions
  if ARGS.debug:
    trainer.extend(print_vmap, trigger=(200, 'iteration'))
  test_iter = C.iterators.SerialIterator(test_data, 128, repeat=False, shuffle=False)
  trainer.extend(T.extensions.Evaluator(test_iter, cmodel, device=-1), name='test', trigger=(10, 'iteration'))
  # trainer.extend(T.extensions.snapshot(filename=fname+'_latest.npz'), trigger=(100, 'iteration'))
  trainer.extend(T.extensions.LogReport(log_name=fname+'_log.json', trigger=(10, 'iteration')))
  trainer.extend(T.extensions.FailOnNonNumber())
  train_keys = ['uloss', 'igloss', 'oloss', 'uacc', 'igacc', 'oacc', 'vloss']
  test_keys = ['uloss', 'oloss', 'uacc', 'oacc']
  trainer.extend(T.extensions.PrintReport(['iteration'] + ['main/'+k for k in train_keys] + ['test/main/'+k for k in test_keys] + ['elapsed_time']))
  # ---------------------------
  print(f"---- FOLD {foldid} ----")
  try:
    trainer.run()
  except KeyboardInterrupt:
    if not ARGS.debug:
      return
  # Save learned invariants
  with open(trainer.out + '/' + fname + '.out', 'w') as f:
    f.write("---- META ----\n")
    train_data = np.stack(train_data)
    test_data = np.stack(test_data)
    meta = {'train': train_data.shape, 'train_tasks': np.unique(train_data[:,0], return_counts=True),
            'test': test_data.shape, 'test_tasks': np.unique(test_data[:,0], return_counts=True),
            'foldid': foldid}
    f.write(str(meta))
    f.write("\n---- INVS ----\n")
    f.write(str(model.inv_examples))
    f.write("\n--------\n")
    f.write(str(model.log['vmap'][0].array))
    for t in range(1, TASKS+1):
      f.write("\n---- SAMPLE ----\n")
      test_data = np.stack(test_data) # (S, 1+L+1)
      np.random.shuffle(test_data)
      batch = test_data[test_data[:, 0] == t][:4] # (B, 1+L+1)
      f.write("Input:\n")
      f.write(str(batch))
      out = model(batch) # (B, V)
      f.write("\nOutput:\n")
      f.write(str(out.array))
      f.write("\nAtt:\n")
      f.write(str(model.log['uniatt'][0].array))
    f.write("\n---- END ----\n")
  if ARGS.debug:
    print(batch)
    import ipdb; ipdb.set_trace()
    out = model(batch)

# ---------------------------

# Training loop
for foldidx, (traind, testd) in enumerate(nfolds):
  # We'll ensure the model sees every symbol at least once in training
  # at test time symbols might appear in different unseen sequences
  vtraind = np.stack(traind)
  if ARGS.tsize > 0:
    # For each task select at most tsize many examples
    vtraind = np.concatenate([vtraind[vtraind[:, 0] == tid][:ARGS.tsize] for tid in range(1, TASKS+1)]) # (<=tsize, 1+L+1)
  train_syms = vtraind[:, 1:-1]
  assert len(np.unique(train_syms)) == VOCAB-1, "Some symbols are missing from training."
  train(vtraind, testd, foldidx)
