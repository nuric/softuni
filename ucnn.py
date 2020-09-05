"""Unification CNN."""
import argparse
import uuid
import json
import pickle
import sys
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T

# Disable scientific printing
np.set_printoptions(threshold=10000, suppress=True, precision=5, linewidth=180)
# pylint: disable=line-too-long

# Arguments
parser = argparse.ArgumentParser(description="Run UCNN on randomly generated tasks.")
# parser.add_argument("task", help="Task name to solve.")
parser.add_argument("--name", help="Name prefix for saving files etc.")
parser.add_argument("-gr", "--grid", nargs='+', default=[3, 3], type=int, help="Size of input grid.")
parser.add_argument("-s", "--symbols", default=8, type=int, help="Number of symbols.")
parser.add_argument("-i", "--invariants", default=1, type=int, help="Number of invariants per task.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-nu", "--nouni", action="store_true", help="Disable unification.")
parser.add_argument("-t", "--train_size", default=0, type=int, help="Training size per task, 0 to use everything.")
parser.add_argument("-g", "--gsize", default=1000, type=int, help="Random data tries per task.")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Training batch size.")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--data", choices=['save', 'load'], help="Save or load generated data.")
ARGS = parser.parse_args()

GRID = np.array(ARGS.grid, dtype=np.int8)
EMBED = ARGS.embed
# We'll add 1, reserve 0 for padding
VOCAB = ARGS.symbols + 1
FOLDS = 5

# ---------------------------

def rand_syms(size, with_padding=False, replace=False):
  """Return unique random symbols."""
  # We'll add 1, reserve 0 for padding
  if with_padding:
    return np.random.choice(ARGS.symbols+1, size=size, replace=replace)
  return np.random.choice(ARGS.symbols, size=size, replace=replace) + 1

def rand_canvas(size: tuple = None, sym_prob: float = 0.1):
  """Return canvas with random symbols at each position."""
  return rand_syms(size or GRID, with_padding=True) * (np.random.random(size or GRID) < sym_prob)

def blank_canvas(size: tuple = None):
  """Return a blank canvas to fill."""
  return np.zeros(size or GRID, dtype=np.int16)

# Generate random data for tasks
def gen_task1() -> np.ndarray:
  """Task 1: 2x2 box of symbols."""
  canv = blank_canvas()
  r, c = np.random.randint(GRID-1, size=2, dtype=np.int8)
  sym = rand_syms(1)[0]
  canv[r:r+2, c:c+2] = sym
  return [1, sym], canv # (2,), (W, H)

def gen_task2() -> np.ndarray:
  """Task 2: top left (head) of vertical diag or horizontal unique symbols."""
  canv = blank_canvas()
  # Let's pick an orientation
  rand = np.random.rand()
  rows, cols = GRID
  length = 3
  syms = rand_syms(length)
  lr = np.arange(length) # length of sequence
  if rand < 0.33:
    # vertical
    r = np.random.randint(rows-length+1, dtype=np.int8)
    c = np.random.randint(cols, dtype=np.int8)
    rows, cols = r+lr, c
  elif rand < 0.66:
    # diagonal
    r = np.random.randint(rows-length+1, dtype=np.int8)
    c = np.random.randint(cols-length+1, dtype=np.int8)
    rows, cols = r+lr, c+lr
  else:
    # horizontal
    r = np.random.randint(rows, dtype=np.int8)
    c = np.random.randint(cols-length+1, dtype=np.int8)
    rows, cols = r, c+lr
  canv[rows, cols] = syms
  return [2, syms[0]], canv

def gen_task3() -> np.ndarray:
  """Task 3: centre of cross or a plus sign."""
  canv = blank_canvas()
  r, c = np.random.randint(GRID-2, size=2, dtype=np.int8)
  # Do we create a cross or a plus sign?
  syms = rand_syms(5) # a 3x3 sign has 2 symbols, outer and centre
  # syms = np.array([syms[0], syms[0], syms[1], syms[0], syms[0]])
  if np.random.rand() < 0.5:
    # Let's do a plus
    rows, cols = [r, r+1, r+1, r+1, r+2], [c+1, c, c+1, c+2, c+1]
  else:
    # Let's do a cross
    rows, cols = [r, r, r+1, r+2, r+2], [c, c+2, c+1, c, c+2]
  canv[rows, cols] = syms
  return [3, syms[2]], canv

def gen_task4() -> np.ndarray:
  """Task 4: main corner of a triangle."""
  canv = blank_canvas()
  r, c = np.random.randint(GRID-2, size=2, dtype=np.int8)
  syms = rand_syms(6) # 6 symbols for triangle
  # Which orientation? We'll create 4
  rand = np.random.rand()
  if rand < 0.25:
    # top left
    rows, cols = [r, r, r, r+1, r+1, r+2], [c, c+1, c+2, c, c+1, c]
  elif rand < 0.50:
    # top right
    rows, cols = [r, r, r, r+1, r+1, r+2], [c+2, c, c+1, c+1, c+2, c+2]
  elif rand < 0.75:
    # bottom left
    rows, cols = [r+2, r, r+1, r+1, r+2, r+2], [c, c, c, c+1, c+1, c+2]
  else:
    # bottom right
    rows, cols = [r+2, r, r+1, r+1, r+2, r+2], [c+2, c+2, c+1, c+2, c, c+1]
  canv[rows, cols] = syms
  return [4, syms[0]], canv

TASKS = 4

def gen_all() -> np.ndarray:
  """Generate all tasks."""
  gdata = list()
  for i in range(1, TASKS+1):
    f = globals()['gen_task'+str(i)]
    for _ in range(ARGS.gsize):
      (taskid, target), grid = f()
      gdata.append(np.concatenate(([taskid], grid.flatten(), [target])))
  gdata = np.unique(np.stack(gdata), axis=0) # (tasks*S, 1+W*H+1)
  np.random.shuffle(gdata)
  return gdata

def print_tasks(batch_tasks: np.ndarray, file=sys.stdout):
  """Pretty print compressed tasks."""
  # batch_tasks (..., 1+W*H+1)
  ts = batch_tasks.reshape((-1, batch_tasks.shape[-1])) # (B, 1+W*H+1)
  task_ids, targets = ts[:, 0], ts[:, -1] # (B, 1)
  canvs = ts[:, 1:-1].reshape((ts.shape[0],) + tuple(GRID)) # (B, W, H)
  for tid, canv, target in zip(task_ids, canvs, targets):
    print("TASK ID:", tid, file=file)
    print("CANVAS:", file=file)
    print(canv, file=file)
    print("TARGET:", target, file=file)

data = gen_all() # (S, 1+W*H+1)
nfolds = C.datasets.get_cross_validation_datasets_random(data, FOLDS) # 5 folds, list of 5 tuples train/test

# ---
# Save or load data
if ARGS.data == "save":
  with open('data/grid_data.pickle', 'wb') as f:
    pickle.dump((data, nfolds), f)
  print("Saved generated data.")
  sys.exit()
if ARGS.data == "load":
  with open('data/grid_data.pickle', 'rb') as f:
    data, nfolds = pickle.load(f)
  print("Loaded pre-generated data.")
# ---

metadata = {'data': data.shape, 'tasks': np.unique(data[:, 0], return_counts=True),
            'folds': len(nfolds), 'train': len(nfolds[0][0]), 'test': len(nfolds[0][1])}
print(metadata)

# ---------------------------

# Unification CNN
class UCNN(C.Chain):
  """Unification convolutional neural network network."""
  def __init__(self, inv_examples):
    super().__init__()
    self.add_persistent('inv_examples', inv_examples) # (T, I, 1+W*H+1)
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(VOCAB, EMBED, ignore_label=0)
      self.vmap_params = C.Parameter(0.0, (inv_examples.shape[:2]) + (VOCAB,), name='vmap_params')
      self.uni_conv1 = L.Convolution2D(EMBED+TASKS, EMBED, ksize=3, stride=1, pad=1)
      self.uni_conv2 = L.Convolution2D(EMBED, EMBED, ksize=3, stride=1, pad=1)
      self.conv1 = L.Convolution2D(EMBED+TASKS, EMBED, ksize=3, stride=1, pad=1)
      self.conv2 = L.Convolution2D(EMBED, EMBED, ksize=3, stride=1, pad=1)
      self.fc1 = L.Linear(EMBED, EMBED)
    self.log = None

  def tolog(self, key, value):
    """Append to log dictionary given key value pair."""
    loglist = self.log.setdefault(key, [])
    loglist.append(value)

  def predict(self, combined_x):
    """Forward pass for combined input."""
    # combined_x (..., W, H, E+T)
    in_x = F.reshape(combined_x, (-1,) + combined_x.shape[-3:]) # (N, W, H, E+T)
    in_x = F.swapaxes(in_x, -1, -3) # (N, E+T, H, W)
    out = F.relu(self.conv1(in_x)) # (N, E, H, W)
    out = F.relu(self.conv2(out)) # (N, E, W', H')
    out = F.max_pooling_2d(out, tuple(GRID)) # (N, E, W', H')
    out = self.fc1(out) # (N, V)
    out = F.squeeze(out) @ self.embed.W.T # (N, V)
    out = F.reshape(out, combined_x.shape[:-3] + (VOCAB,)) # (..., V)
    return out

  def embed_predict(self, examples):
    """Just a forward prediction of given example."""
    # examples (..., 1+W*H)
    ex = F.reshape(examples[..., 1:], examples.shape[:-1] + tuple(GRID)) # (..., W, H)
    ex = self.embed(ex) # (..., W, H, E)
    task_id = F.embed_id(examples[..., 0]-1, np.eye(TASKS, dtype=np.float32)) # (..., T)
    task_id = F.tile(task_id[..., None, None, :], ex.shape[-3:-1] + (1,)) # (..., W, H, T)
    combined_ex = F.concat((ex, task_id), axis=-1) # (..., W, H, E+T)
    return self.predict(combined_ex) # (..., V)

  def compute_ground_loss(self, examples, log_prefix=''):
    """Compute loss and accuracy on ground examples."""
    # examples (..., 1+W*H+1)
    preds = self.embed_predict(examples[..., :-1]) # (..., V)
    preds = F.reshape(preds, (-1, VOCAB)) # (N, V)
    targets = F.flatten(examples[..., -1]) # (N,)
    loss = F.softmax_cross_entropy(preds, targets) # ()
    acc = F.accuracy(preds, targets) # ()
    self.tolog(log_prefix+'loss', loss)
    self.tolog(log_prefix+'acc', acc)
    return preds

  def forward(self, ground_examples: np.ndarray):
    """Compute the forward inference pass for given stories."""
    # ground_examples (B, 1+W*H+1)
    self.log = dict()
    # ---------------------------
    # Invariant ground prediction
    self.compute_ground_loss(self.inv_examples, log_prefix='ig')
    # Ground example prediction
    self.compute_ground_loss(ground_examples, log_prefix='o')
    # ---------------------------
    # Unification case
    task_ids = ground_examples[:, 0] # (B,)
    ground_inputs = ground_examples[:, 1:-1] # (B, W*H)

    invs_inputs = self.inv_examples[..., 1:-1] # (T, I, W*H)
    # invs_inputs = invariant_inputs[task_ids-1] # (B, I, W*H)

    # Embed ground examples
    eg = self.embed(ground_inputs) # (B, W*H, E)
    ei = self.embed(invs_inputs) # (T, I, W*H, E)

    # Extract unification features
    tids = F.embed_id(task_ids-1, np.eye(TASKS, dtype=np.float32)) # (B, T)
    tids = F.repeat(tids[:, None], eg.shape[1], 1) # (B, W*H, T)
    itids = np.eye(TASKS, dtype=np.float32) # (T, T)
    itids = F.tile(itids[:, None, None, :], (1, invs_inputs.shape[1], invs_inputs.shape[2], 1)) # (T, I, W*H, T)

    egt = F.concat((eg, tids), -1) # (B, W*H, E+T)
    eit = F.concat((ei, itids), -1) # (T, I, W*H, E+T)
    egt = F.reshape(egt, egt.shape[:1] + tuple(GRID) + egt.shape[-1:]) # (B, W, H, E+T)
    eit = F.reshape(eit, (-1,) + tuple(GRID) + eit.shape[-1:]) # (T*I, W, H, E+T)
    egt = F.swapaxes(egt, -1, -3) # (B, E+T, W, H)
    eit = F.swapaxes(eit, -1, -3) # (T*I, E+T, W, H)

    gfeats = F.relu(self.uni_conv1(egt)) # (B, E, W, H)
    ifeats = F.relu(self.uni_conv1(eit)) # (T*I, E, W, H)
    gfeats = self.uni_conv2(gfeats) # (B, E, W, H)
    ifeats = self.uni_conv2(ifeats) # (T*I, E, W, H)
    gfeats = F.reshape(gfeats, gfeats.shape[:2] + (-1,)) # (B, E, W*H)
    ifeats = F.reshape(ifeats, ei.shape[:2] + ifeats.shape[1:2] + (-1,)) # (T, I, E, W*H)

    batch_ifeats = ifeats[task_ids-1] # (B, I, E, W*H)
    # (B, I, E, W*H) x (B, E, W*H) -> (B, I, W*H, W*H)
    uni_att = F.einsum("ijek,iel->ijkl", batch_ifeats, gfeats) # (B, I, W*H, W*H)
    mask = -100*(ground_inputs == 0) # (B, W*H) cannot attend to padding
    uni_att += mask[:, None, None] # (B, I, W*H, W*H)
    uni_att = F.softmax(uni_att, axis=-1) # (B, I, W*H, W*H)
    self.tolog('uniatt', uni_att)

    # (B, I, W*H, W*H) x (B, W*H, E) -> (B, I, W*H, E)
    eu = F.einsum("ijkl,ile->ijke", uni_att, eg) # (B, I, W*H, E)

    # Compute variable map
    vmap = F.sigmoid(self.vmap_params*10) # (T, I, V)
    mask = np.ones(VOCAB) # (V,)
    mask[0] = 0 # padding symbol cannot be variable
    vmap *= mask # (T, I, V)
    self.tolog('vmap', vmap)
    vmap = vmap[np.arange(vmap.shape[0])[:, None, None], np.arange(vmap.shape[1])[None, :, None], invs_inputs] # (T, I, W*H)
    vmap = vmap[task_ids-1] # (B, I, W*H)

    batch_ei = ei[task_ids-1] # (B, I, W*H, E)
    uni_embed = (vmap[..., None]*eu + (1-vmap)[..., None]*batch_ei) # (B, I, W*H, E)

    # Make the prediction on the unification
    batch_itids = itids[task_ids-1] # (B, I, W*H, T)
    uni_embed = F.concat((uni_embed, batch_itids), -1) # (B, I, W*H, E+T)
    uni_inputs = F.reshape(uni_embed, uni_embed.shape[:2] + tuple(GRID) + uni_embed.shape[-1:]) # (B, I, W, H, E+T)
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

  def forward(self, ground_examples: np.ndarray):
    """Compute total loss to train."""
    # ground_examples (B, 1+W*H+1)
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
    # for k in ['o']:
      report[k+'loss'] = self.predictor.log[k+'loss'][0]
      report[k+'acc'] = self.predictor.log[k+'acc'][0]
    C.report(report, self)
    # return report['oloss']

    vloss = F.sum(self.predictor.vmap_params) # ()
    report['vloss'] = vloss
    # ---------------------------
    C.report(report, self)
    return self.uniparam*(uloss + 0.1*vloss + report['igloss']) + (1-self.uniparam)*report['oloss']

# ---------------------------

# Training tools
def select_invariants(data: list, taskid: int):
  """Select I many examples with different answers."""
  data = np.stack(data) # (S, 1+W*H+1)
  np.random.shuffle(data)
  invs = data[data[:, 0] == taskid][:ARGS.invariants] # (<=I, 1+W*H+1)
  # Check if we have enough, tile if not
  if invs.shape[0] < ARGS.invariants:
    invs = np.tile(invs, ((ARGS.invariants//invs.shape[0])+1, 1)) # (>=I, 1+W*H+1)
    invs = invs[:ARGS.invariants]
  return invs

def print_vmap(trainer):
  """Enable unification loss function in model."""
  print_tasks(trainer.updater.get_optimizer('main').target.predictor.inv_examples)
  print(trainer.updater.get_optimizer('main').target.predictor.log['vmap'])

# ---------------------------

# Training on single fold
def train(train_data, test_data, foldid: int = 0):
  """Train new UMLP on given data."""
  # Setup invariant repositories
  # we'll take I many examples for each task with different answers for each fold
  invariants = [select_invariants(train_data, i) for i in range(1, 5)] # T x (I, 1+W*H+1)
  invariants = np.stack(invariants) # (T, I, 1+W*H+1)
  # ---------------------------
  # Setup model
  model = UCNN(invariants)
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam(alpha=ARGS.learning_rate).setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  updater = T.StandardUpdater(train_iter, optimiser, device=-1)
  trainer = T.Trainer(updater, (2000, 'iteration'), out='results/ucnn_result')
  # ---------------------------
  fname = (ARGS.name.format(foldid=foldid) if ARGS.name else '') or ('debug' if ARGS.debug else '') or str(uuid.uuid4())
  # Setup trainer extensions
  if ARGS.debug:
    trainer.extend(print_vmap, trigger=(1000, 'iteration'))
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
  # Save run parameters
  params = ['symbols', 'invariants', 'embed', 'train_size', 'learning_rate', 'nouni', 'batch_size']
  params = {k: vars(ARGS)[k] for k in params}
  params['name'] = fname
  params['foldid'] = foldid
  with open(trainer.out + '/' + fname + '_params.json', 'w') as f:
    json.dump(params, f)
  # Save learned invariants
  with open(trainer.out + '/' + fname + '.out', 'w') as f:
    f.write("---- META ----\n")
    train_data = np.stack(train_data)
    test_data = np.stack(test_data)
    meta = {'train': train_data.shape, 'train_tasks': np.unique(train_data[:,0], return_counts=True),
            'test': test_data.shape, 'test_tasks': np.unique(test_data[:,0], return_counts=True),
            'foldid': foldid}
    f.write(str(meta))
    f.write("\n--------\n")
    for t in range(1, TASKS+1):
      f.write(f"\n---- SAMPLE {t}----\n")
      test_data = np.stack(test_data) # (S, 1+W*H+1)
      np.random.shuffle(test_data)
      batch = test_data[test_data[:, 0] == t][:4] # (B, 1+W*H+1)
      f.write("Input:\n")
      print_tasks(batch, file=f)
      out = model(batch) # (B, V)
      f.write("\nOutput:\n")
      f.write(np.array_str(out.array))
      uniatt = model.log['uniatt'][0].array # (B, I, W*H, W*H)
      for i in range(uniatt.shape[0]):
        for j in range(uniatt.shape[1]):
          f.write(f"\nAtt Input {i} with Inv {j}:\n")
          ut = uniatt[i,j] # (W*H, W*H)
          inv = model.inv_examples[t-1, j, 1:-1] # (W*H)
          toprint = ut[np.nonzero(inv)] # (nonzero, W*H)
          toprint = toprint.reshape((-1,) + tuple(GRID))
          f.write(np.array_str(toprint))
      f.write("\nInvs:\n")
      print_tasks(model.inv_examples[t-1], file=f)
      f.write("\nVmap:\n")
      f.write(np.array_str(model.log['vmap'][0].array[t-1]))
    f.write("\n---- END ----\n")
  if ARGS.debug:
    for testd in test_data:
      preds = model(testd[None, :])
      if np.argmax(preds.array) != testd[-1]:
        print_tasks(testd)
        # print(model.log)
        print(preds)
        print(np.argmax(preds.array))
        import ipdb; ipdb.set_trace()
        print("HERE")

# ---------------------------

# Training loop
for foldidx, (traind, testd) in enumerate(nfolds):
  # We'll ensure the model sees every symbol at least once in training
  # at test time symbols might appear in different unseen sequences
  vtraind = np.stack(traind)
  if ARGS.train_size > 0:
    # For each task select at most tsize many examples
    vtraind = np.concatenate([vtraind[vtraind[:, 0] == tid][:ARGS.train_size] for tid in range(1, TASKS+1)]) # (<=tsize, 1+W*H+1)
  train_syms = vtraind[:, 1:-1]
  assert len(np.unique(train_syms)) == VOCAB, "Some symbols are missing from training."
  train(vtraind, testd, foldidx)
