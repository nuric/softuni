"""Unification RNN."""
import argparse
import json
import re
import string
import sys
from functools import partial
import numpy as np
import pandas as pd
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180, threshold=1000000)
# pylint: disable=line-too-long

# Arguments
parser = argparse.ArgumentParser(description="Run URNN on reviews.")
parser.add_argument("name", help="Name prefix for saving files etc.")
parser.add_argument("-l", "--length", default=20, type=int, help="Max length of reviews.")
parser.add_argument("-i", "--invariants", default=1, type=int, help="Number of invariants per task.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-nu", "--nouni", action="store_true", help="Disable unification.")
parser.add_argument("--train_size", default=0, type=int, help="Training size per label, 0 to use everything.")
parser.add_argument("--test_size", default=0, type=int, help="Test size per label, 0 to use everything.")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Training batch size.")
parser.add_argument("-o", "--outf", default="{name}_l{length}_i{invariants}_e{embed}_t{train_size}_f{foldid}")
ARGS = parser.parse_args()

EMBED = ARGS.embed
FOLDS = 5

# ---------------------------

def prep_dataset(dset):
  """Filter dataset and return a new one."""
  # pad=0, start=1, end=2, oov=3
  texts, labels = list(), list()
  for t, l in dset:
    # +2 for start=1 and end=2 tokens
    # if len(t) + 2 > ARGS.length:
      # continue
    t = t[-(ARGS.length-2):]
    t = np.array([1] + [w+3 if w <= ARGS.symbols else 3 for w in t] + [2]).astype(np.int32)
    texts.append(t)
    labels.append(l)
  idxs = np.random.permutation(len(texts))
  texts = np.array(texts)[idxs]
  labels = np.array(labels).astype(np.int8)[idxs]
  return C.datasets.TupleDataset(texts, labels)

def filter_per_label(dataset, size: int):
  """Filter a dataset based to have size per label."""
  if size == 0:
    return dataset
  # Filter by finding size many matching examples
  # Not the most efficient implementation, but C.datasets
  # has an indexing problem with SubDataset, it converts
  # numpy data to list of tuples
  labels = np.stack([dp[1] for dp in dataset])  # (S,)
  texts = np.array([dp[0] for dp in dataset])  # (S,)
  idxs = list()
  for l in np.unique(labels):
    idxs.extend(np.flatnonzero(l == labels)[:size])
  np.random.shuffle(idxs)
  return C.datasets.TupleDataset(texts[idxs], labels[idxs])

def data_stats(dset):
  """Collect dataset statistics."""
  ls = [len(t) for t, _ in dset]
  vocab = np.unique([w for t, _ in dset for w in t])
  lstats = np.unique([l for _, l in dset], return_counts=True)[1]
  stats = {'total': len(ls), 'vocab': len(vocab), 'maxlen': max(ls), 'labels': lstats}
  return stats

word2idx = dict()
def encode_sent(encode: bool, sent: str):
  """Encode given sentence."""
  s = sent.translate(str.maketrans('', '', string.punctuation))
  s = s.strip().lower()
  s = re.sub(' +', ' ', s)
  ws = s.split(' ')
  if encode:
    ws = np.array([word2idx.setdefault(w, len(word2idx)+1) for w in ws], dtype=np.int32)
  return ws
word2idx['PAD'] = 0

# Load Sentiment Reviews
sent_labels = pd.read_csv('data/stanfordSentimentTreebank/sentiment_labels.txt', sep='|', header=0, names=['label'], index_col=0)
sent_labels = sent_labels[((sent_labels.label > 0.9) | (sent_labels.label < 0.1))]
sent_labels['label'] = np.where(sent_labels.label < 0.1, 0, 1)
phrases = pd.read_csv('data/stanfordSentimentTreebank/dictionary.txt', sep='|', header=None, names=['phrase'], index_col=1)
df = sent_labels.join(phrases)
df['enc_phrase'] = df['phrase'].apply(partial(encode_sent, True))
df['proc_phrase'] = df['phrase'].apply(partial(encode_sent, False))
df['length'] = df['enc_phrase'].apply(len)
# Remove short phrases
df = df[(df.length > 3) & (df.length < ARGS.length)]
print(df)
data = C.datasets.TupleDataset(df['enc_phrase'].to_numpy(), df['label'].to_numpy())

# Load IMDB data
# Using https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
# Download from https://s3.amazonaws.com/text-datasets/imdb.npz
# And https://s3.amazonaws.com/text-datasets/imdb_word_index.json
# with np.load('data/imdb/imdb.npz', allow_pickle=True) as f:
  # traind = C.datasets.TupleDataset(f['x_train'], f['y_train'])
  # testd = C.datasets.TupleDataset(f['x_test'], f['y_test'])
  # traind = prep_dataset(traind)
  # testd = prep_dataset(testd)
  # testd = filter_per_label(testd, ARGS.test_size)

# with open('data/imdb/imdb_word_index.json') as f:
  # word2idx = json.load(f)
  # # +3 for start, end and oov tokens
  # # the indexing starts from 1
  # word2idx = {k: v+3 for k, v in word2idx.items()}
  # word2idx['PAD'] = 0
  # word2idx['START'] = 1
  # word2idx['END'] = 2
  # word2idx['OOV'] = 3
idx2word = {v:k for k, v in word2idx.items()}

def print_tasks(in_data, file=sys.stdout):
  """Print task."""
  if isinstance(in_data, tuple):
    in_data = zip(*in_data)
  for t, l in in_data:
    s = [idx2word[w] for w in t]
    print(s, '->', l, file=file)

nfolds = C.datasets.get_cross_validation_datasets_random(data, FOLDS) # 5 folds, list of 5 tuples train/test
# Filter per label
nfolds = [(filter_per_label(td, ARGS.train_size), filter_per_label(vd, ARGS.test_size)) for td, vd in nfolds]

metadata = {'data': data_stats(data), 'folds': len(nfolds)}
for foldidx, (trainfold, testfold) in enumerate(nfolds):
  metadata['train' + str(foldidx)] = data_stats(trainfold)
  metadata['test' + str(foldidx)] = data_stats(testfold)
print(metadata)

# ---------------------------

def sequence_embed(embed, xs):
  """Embed sequences of integers."""
  # xt [(L1,), (L2,), ...]
  xs = list(xs)  # Chainer quirk expects lists
  x_len = [len(x) for x in xs]
  x_section = np.cumsum(x_len[:-1])
  ex = embed(F.concat(xs, axis=0))
  exs = F.split_axis(ex, x_section, 0)
  return exs

# ---------------------------

# Unification Network
class URNN(C.Chain):
  """Unification RNN network for classification."""
  def __init__(self, inv_examples):
    super().__init__()
    self.inv_examples = inv_examples
    inv_texts, inv_labels = inv_examples
    maxilen = max([len(t) for t in inv_texts])
    self.add_persistent('inv_texts', inv_texts) # (I,)
    self.add_persistent('inv_labels', inv_labels) # (I,)
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx)+1, 32, ignore_label=0)
      self.vmap_params = C.Parameter(0.0, (len(inv_texts), maxilen), name='vmap_params')
      self.bilstm = L.NStepBiLSTM(1, 32, 32, 0)
      self.uni_linear = L.Linear(32*2, 32)
      self.fc1 = L.Linear(32*2, 1)

  def predict(self, embed_seqs):
    """Predict class on embeeded seqs."""
    # embed_seqs B x [(L1, E), (L2, E), ...]
    hy, _, ys = self.bilstm(None, None, embed_seqs)  # (2, B, E), _, B x [(L1, E), ...]
    hy = F.transpose(hy, [1, 0, 2])  # (B, 2, E)
    hy = F.reshape(hy, [hy.shape[0], -1])  # (B, 2*E)
    hy = F.dropout(hy, 0.5)  # (B, 2*E)
    pred = self.fc1(hy)  # (B, 1)
    return ys, pred

  def forward(self, texts):
    """Compute the forward inference pass for given stories."""
    # texts [(L1,), (L2,), (L3,)]
    report = dict()
    # ---------------------------
    # Ground example prediction
    oe = sequence_embed(self.embed, texts)  # B x [(L1, E), (L2, E), ...]
    oys, opred = self.predict(oe)  # B x [(L1, E), ...], (B, 1)
    report['opred'] = opred
    # Invariant ground prediction
    ie = sequence_embed(self.embed, self.inv_texts)  # I x ([L1, E), ...]
    iys, ipred = self.predict(ie)  # I x [(L1, E), ..], (I, 1)
    report['igpred'] = ipred
    # ---------------------------
    # Extract unification features
    iufeats = F.pad_sequence(iys)  # (I, LI, E)
    oufeats = F.pad_sequence(oys)  # (B, LB, E)
    iufeats = self.uni_linear(iufeats, n_batch_axes=2)  # (I, LI, E)
    oufeats = self.uni_linear(oufeats, n_batch_axes=2)  # (B, LB, E)
    # ---------------------------
    # Unification attention
    # (B, LB, E) x (I, LI, E) -> (B, I, LI, LB)
    uniatt = F.einsum('ble,ife->bifl', oufeats, iufeats)
    # Mask to stop attention to padding
    padded_texts = F.pad_sequence(list(texts)).array  # (B, LB)
    mask = -100*(padded_texts == 0)  # (B, LB)
    # Compute attention
    uniatt += mask[:, None, None]  # (B, I, LI, LB)
    uniatt = F.softmax(uniatt, -1)  # (B, I, LI, LB)
    report['uniatt'] = uniatt
    # ---------------------------
    # Compute variable map
    vmap = F.sigmoid(self.vmap_params*10) # (I, LI)
    # Mask out padding
    padded_invs = F.pad_sequence(list(self.inv_texts)).array  # (I, LI)
    mask = (padded_invs != 0)  # (I, LI)
    # Mask out vmap
    vmap *= mask
    report['vmap'] = vmap
    # ---------------------------
    # Compute unified representation
    padded_oe = F.pad_sequence(oe)  # (B, LB, E)
    padded_ie = F.pad_sequence(ie)  # (I, LI, E)
    # (B, I, LI, LB) x (B, LB, E) -> (B, I, LI, E)
    ue = F.einsum('bilf,bfe->bile', uniatt, padded_oe)
    ue = vmap[..., None]*ue + (1-vmap[..., None])*padded_ie  # (B, I, LI, E)
    ue = F.reshape(ue, (-1,) + ue.shape[2:])  # (B*I, LI, E)
    ue = F.separate(ue, 0)  # B*I x [(LI, E), ...]
    # ---------------------------
    # Compute unification predictions
    _, upred = self.predict(ue)  # (B*I, 1)
    upred = F.reshape(upred, uniatt.shape[:2] + (1,))  # (B, I, 1)
    upred = F.sum(upred, 1)  # (B, 1)
    report['upred'] = upred
    # ---------------------------
    return report

# Wrapper chain for training
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    # self.add_persistent('uniparam', not ARGS.nouni)
    self.add_persistent('uniparam', 0.5)
    with self.init_scope():
      self.predictor = predictor

  def forward(self, texts, labels):
    """Compute total loss to train."""
    # texts [(L1,), (L2,), (L3,)]
    # labels (B,)
    report = dict()
    r = self.predictor(texts)
    # ---------------------------
    # Compute loss and accs
    labels = labels[:, None]  # (B, 1)
    inv_labels = self.predictor.inv_labels[:, None]  # (I, 1)
    for k, t in [('o', labels), ('ig', inv_labels), ('u', labels)]:
      report[k + 'loss'] = F.sigmoid_cross_entropy(r[k + 'pred'], t)
      report[k + 'acc'] = F.binary_accuracy(r[k + 'pred'], t)
    # ---------------------------
    # Aux lossess
    vloss = F.sum(self.predictor.vmap_params) # ()
    report['vloss'] = vloss
    # ---------------------------
    C.report(report, self)
    return self.uniparam*(report['uloss'] + 0.1*report['vloss'] + report['igloss']) + (1-self.uniparam)*report['oloss']

# ---------------------------

def print_vmap(trainer):
  """Enable unification loss function in model."""
  print_tasks(trainer.updater.get_optimizer('main').target.predictor.inv_examples)
  print(F.sigmoid(trainer.updater.get_optimizer('main').target.predictor.vmap_params*10))

def converter(batch, _):
  """Curate a batch of samples."""
  # B x [((L1,) True), ((L2,), False), ...]
  texts = np.array([dp[0] for dp in batch])  # (B,)
  labels = np.stack([dp[1] for dp in batch])  # (B,)
  return texts, labels

# ---------------------------

# Training on single fold
def train(train_data, test_data, foldid: int = 0):
  """Train new UMLP on given data."""
  # Setup invariant repositories
  # we'll take I many examples at random
  idxs = np.random.choice(len(train_data), size=ARGS.invariants, replace=False)
  invariants = train_data[idxs]
  # ---------------------------
  # Setup model
  model = URNN(invariants)
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam().setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
  trainer = T.Trainer(updater, (2000, 'iteration'), out='urnn_result')
  # ---------------------------
  fname = ARGS.outf.format(**vars(ARGS), foldid=foldid)
  # Setup trainer extensions
  if ARGS.debug:
    trainer.extend(print_vmap, trigger=(200, 'iteration'))
  test_iter = C.iterators.SerialIterator(test_data, ARGS.batch_size, repeat=False, shuffle=False)
  trainer.extend(T.extensions.Evaluator(test_iter, cmodel, converter=converter, device=-1), name='test', trigger=(10, 'iteration'))
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
  # ---------------------------
  # Save learned invariants
  test_iter.reset()
  test_texts, test_labels = converter(next(test_iter), None)
  test_texts, test_labels = test_texts[:4], test_labels[:4]
  out = {k: v.array for k, v in model(test_texts).items()}
  with open(trainer.out + '/' + fname + '.out', 'w') as f:
    f.write("---- META ----\n")
    metadata['foldid'] = foldid
    f.write(str(metadata))
    f.write("\n---- INVS ----\n")
    print_tasks(model.inv_examples, file=f)
    f.write("\n--------\n")
    f.write(np.array_str(out['vmap']))
    f.write("\n---- SAMPLE ----\n")
    f.write("Input:\n")
    print_tasks((test_texts, test_labels), file=f)
    f.write("\nOutput:\n")
    f.write(np.array_str(out['upred']))
    f.write("\nAtt:\n")
    idxs = np.argsort(out['uniatt'], -1)[..., ::-1]
    idxs = idxs[..., :3]
    f.write(np.array_str(idxs))
    f.write(np.array_str(np.take_along_axis(out['uniatt'], idxs, axis=-1)))
    f.write("\n---- END ----\n")
  if ARGS.debug:
    print_tasks((test_texts, test_labels))
    import ipdb; ipdb.set_trace()
    out = model(test_texts)

# ---------------------------

# Training loop
for foldidx, (trainfold, testfold) in enumerate(nfolds):
  train(trainfold, testfold, foldidx)
