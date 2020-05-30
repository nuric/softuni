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
  s = sent.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
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
df = df[(df.length > 3) & (df.length <= ARGS.length)]
print(df)
data = C.datasets.TupleDataset(df['enc_phrase'].to_numpy(), df['label'].to_numpy())
idx2word = {v:k for k, v in word2idx.items()}

# Load word embeddings
print("Loading word embeddings.")
wordembeds = np.zeros((len(word2idx)+1, 300), dtype=np.float32)
word_count = 0
with open('data/numberbatch-en-19.08.txt') as f:
  for i, l in enumerate(f):
    if i == 0:
      continue  # skip first line
    word, *enc = l.split(' ')
    if word in word2idx:
      word_count += 1
      wordembeds[word2idx[word]] = np.array(enc)
print(f"Loaded {word_count} many vectors out of {len(word2idx)}.")


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

# ---------------------------

# Unification Network
class URNN(C.Chain):
  """Unification RNN network for classification."""
  def __init__(self):
    super().__init__()
    # Create model parameters
    with self.init_scope():
      # self.embed = L.EmbedID(len(word2idx)+1, 32, ignore_label=0)
      # self.uni_embed = L.EmbedID(len(word2idx)+1, 32, ignore_label=0)
      self.embed = L.Linear(300, 16)
      self.uni_embed = L.Linear(16, 16)
      self.var_linear = L.Linear(16, 1)
      self.bigru_state = C.Parameter(0.0, (1, 1, 16), name='bigru_state')
      self.bigru = L.NStepGRU(1, 16, 16, 0)
      # self.uni_bigru = L.NStepBiGRU(1, 32, 32, 0)
      # self.uni_linear = L.Linear(32*2, 32)
      self.fc1 = L.Linear(16*1, 1)

  def predict(self, embed_seqs):
    """Predict class on embeeded seqs."""
    # embed_seqs B x [(L1, E), (L2, E), ...]
    init_state = F.tile(self.bigru_state, (1, len(embed_seqs), 1))  # (2, B, E)
    hy, ys = self.bigru(init_state, embed_seqs)  # (2, B, E), B x [(L1, E), ...]
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
    def sequence_embed(xs):
      """Embed sequences of integers."""
      # xt [(L1,), (L2,), ...]
      xs = list(xs)  # Chainer quirk expects lists
      x_len = [len(x) for x in xs]
      x_section = np.cumsum(x_len[:-1])
      x_concat = F.concat(xs, axis=0)  # (L1+L2...,)
      # ex = self.embed(x_concat) # (..., E)
      ex = F.embed_id(x_concat, wordembeds, ignore_label=0)
      ex = F.tanh(self.embed(ex)) # (..., E)
      uex = self.uni_embed(ex)  # (..., E)
      uvx = self.var_linear(ex)  # (..., 1)
      uvx = F.sigmoid(F.squeeze(uvx, -1))  # (..., )
      # evx = F.concat([ex, uvx[:, None]], -1)  # (..., E+1)
      evxs = F.split_axis(ex, x_section, 0)
      uexs = F.split_axis(uex, x_section, 0)
      uvs = F.split_axis(uvx, x_section, 0)
      return evxs, uexs, uvs
    # Ground example prediction
    ove, ue, uv = sequence_embed(texts)  # B x [(L1, E), (L2, E), ...], Bx[(L1, E), ...], B x [(L1,), (L2,), ...]
    oys, opred = self.predict(ove)  # B x [(L1, E), ...], (B, 1)
    report['opred'] = opred
    # ---------------------------
    # Compute padding mask
    padded_texts = F.pad_sequence(list(texts)).array  # (B, LB)
    mask = -100*(padded_texts == 0)  # (B, LB)
    # ---------------------------
    # Extract unification features
    oufeats = F.pad_sequence(ue)  # (B, LB, E)
    ouvar = F.pad_sequence(uv)  # (B, LB)
    report['vmap'] = ouvar
    # ---------------------------
    # Pick a random example to unify with
    uidxs = np.random.permutation(len(texts))  # (B,)
    while np.any(uidxs == np.arange(len(texts))):
      uidxs = np.random.permutation(len(texts))  # (B,)
    report['uidxs'] = uidxs
    # ---------------------------
    # Unification attention
    # (I, LB, E) x (B, LB, E) -> (I, LB, LB)
    uniatt = F.einsum('ile,ife->ilf', oufeats[uidxs], oufeats)
    # Mask to stop attention to padding
    uniatt += mask[:, None]  # (I, LB, LB)
    uniatt = F.softmax(uniatt, -1)  # (I, LB, LB)
    uniatt *= (padded_texts != 0)[uidxs, ..., None]  # (I, LB, LB)
    report['uniatt'] = uniatt
    # ---------------------------
    # Compute unified representation
    padded_ove = F.pad_sequence(ove)  # (B, LB, E)
    padded_ive = padded_ove[uidxs]  # (I, LI, E)
    ouvar = ouvar[uidxs]  # (I, LI)
    # (I, LB, LB) x (B, LB, E) -> (I, LB, E)
    uve = F.einsum('ilf,ife->ile', uniatt, padded_ove)
    # ---
    uve = ouvar[..., None] * uve + (1-ouvar[..., None]) * padded_ive  # (I, LB, E)
    # Variableness of a rule does not get unified
    # uve = F.concat([uve[..., :-1], padded_ive[..., -1:]], -1)  # (I, LB, E+1)
    uve = F.separate(uve, 0)  # I x [(LB, E), ...]
    ulens = np.array([len(t) for t in texts])[uidxs]  # (I,)
    uve = [seq[:l] for seq, l in zip(uve, ulens)]  # I x [(L1, E), (L2, E), ..]
    # ---------------------------
    # Compute unification predictions
    _, upred = self.predict(uve)  # (I, 1)
    report['upred'] = upred
    # ---------------------------
    return report

# Wrapper chain for training
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    self.add_persistent('uniparam', not ARGS.nouni)
    # self.add_persistent('uniparam', 0.5)
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
    for k, t in [('o', labels), ('u', labels)]:
      report[k + 'loss'] = F.sigmoid_cross_entropy(r[k + 'pred'], t)
      report[k + 'acc'] = F.binary_accuracy(r[k + 'pred'], t)
    # ---------------------------
    # Aux lossess
    vloss = F.mean(r['vmap']) # ()
    report['vloss'] = vloss
    # ---------------------------
    C.report(report, self)
    # return self.uniparam*(report['uloss'] + 0.001*report['vloss']) + (1-self.uniparam)*report['oloss']
    return (report['uloss'] + 0*report['vloss']) + report['oloss']

# ---------------------------

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
  # ---------------------------
  # Setup model
  model = URNN()
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam().setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  test_iter = C.iterators.SerialIterator(test_data, ARGS.batch_size, repeat=False, shuffle=False)
  updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
  trainer = T.Trainer(updater, (2000, 'iteration'), out='urnn_result')
  # ---------------------------
  # Setup debug output
  test_iter.reset()
  debug_texts, debug_labels = converter(next(test_iter), None)
  debug_texts, debug_labels = debug_texts[:4], debug_labels[:4]
  def print_vmap(trainer):
    """Enable unification loss function in model."""
    print_tasks((debug_texts, debug_labels))
    print(model(debug_texts))
  # ---------------------------
  fname = ARGS.outf.format(**vars(ARGS), foldid=foldid)
  # Setup trainer extensions
  if ARGS.debug:
    trainer.extend(print_vmap, trigger=(200, 'iteration'))
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
  out = {k: v if isinstance(v, np.ndarray) else v.array for k, v in model(debug_texts).items()}
  with open(trainer.out + '/' + fname + '.out', 'w') as f:
    f.write("---- META ----\n")
    metadata['foldid'] = foldid
    f.write(str(metadata))
    f.write("\n---- SAMPLE ----\n")
    f.write("Input:\n")
    print_tasks((debug_texts, debug_labels), file=f)
    for k, v in out.items():
      f.write(f"\n{k}:\n")
      f.write(np.array_str(v))
    f.write("\n---- END ----\n")
  if ARGS.debug:
    print_tasks((debug_texts, debug_labels))
    import ipdb; ipdb.set_trace()
    out = model(debug_texts)

# ---------------------------

# Training loop
for foldidx, (trainfold, testfold) in enumerate(nfolds):
  ARGS.nouni = False
  train(trainfold, testfold, foldidx)
  ARGS.nouni = True
  train(trainfold, testfold, foldidx)
  break
