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
parser.add_argument("-e", "--embed", default=16, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-nu", "--nouni", action="store_true", help="Disable unification.")
parser.add_argument("-t", "--train_size", default=0, type=int, help="Training size per label, 0 to use everything.")
parser.add_argument("--test_size", default=0, type=int, help="Test size per label, 0 to use everything.")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Training batch size.")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("-o", "--outf", default="{name}_l{length}_i{invariants}_e{embed}_t{train_size}_f{foldid}")
ARGS = parser.parse_args()

LABEL_T = 0.1 # Lower bound below which is set to 0 
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
sent_labels = sent_labels[((sent_labels.label > (0.5-LABEL_T)+0.5) | (sent_labels.label < LABEL_T))]
sent_labels['label'] = np.where(sent_labels.label < LABEL_T, 0, 1)
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

# Unification Network
class URNN(C.Chain):
  """Unification RNN network for classification."""
  def __init__(self, inv_examples):
    super().__init__()
    self.inv_examples = inv_examples
    # Create model parameters
    with self.init_scope():
      # self.embed = L.EmbedID(len(word2idx)+1, 32, ignore_label=0)
      # self.uni_embed = L.EmbedID(len(word2idx)+1, 32, ignore_label=0)
      self.embed = L.Linear(300, EMBED)
      self.uni_embed = L.Linear(EMBED, EMBED)
      self.var_linear = L.Linear(EMBED, 1)
      self.lstm = L.NStepLSTM(1, EMBED, EMBED, 0)
      self.fc1 = L.Linear(EMBED*1, 1)

  def predict(self, embed_seqs):
    """Predict class on embeeded seqs."""
    # embed_seqs B x [(L1, E), (L2, E), ...]
    hy, _, ys = self.lstm(None, None, embed_seqs)  # (2, B, E), B x [(L1, E), ...]
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
    # Invariant example prediction
    ive, iue, iuv = sequence_embed(self.inv_examples[0]) # I x [(L1, E), ...] ...
    iys, ipred = self.predict(ive) # I x [(L1, E), ...], (I, 1)
    report['igpred'] = ipred
    # ---------------------------
    # Compute padding mask
    padded_texts = F.pad_sequence(list(texts)).array  # (B, LB)
    mask = -100*(padded_texts == 0)  # (B, LB)
    padded_itexts = F.pad_sequence(list(self.inv_examples[0])).array # (I, LI)
    # ---------------------------
    # Extract unification features
    oufeats = F.pad_sequence(ue)  # (B, LB, E)
    iufeats = F.pad_sequence(iue) # (I, LI, E)
    iuvar = F.pad_sequence(iuv)  # (I, LI)
    report['vmap'] = iuvar
    # ---------------------------
    # Unification attention
    # (I, LI, E) x (B, LB, E) -> (B, I, LI, LB)
    uniatt = F.einsum('ile,bfe->bilf', iufeats, oufeats)
    # Mask to stop attention to padding
    uniatt += mask[:, None, None]  # (B, I, LI, LB)
    uniatt = F.softmax(uniatt, -1)  # (B, I, LI, LB)
    uniatt *= (padded_itexts != 0)[..., None] # (B, I, LI, LB)
    report['uniatt'] = uniatt
    # ---------------------------
    # Compute unified representation
    padded_ove = F.pad_sequence(ove)  # (B, LB, E)
    padded_ive = F.pad_sequence(ive)  # (I, LI, E)
    # (B, I, LI, LB) x (B, LB, E) -> (B, I, LI, E)
    uve = F.einsum('bilf,bfe->bile', uniatt, padded_ove)
    # ---
    uve = iuvar[..., None] * uve + (1-iuvar[..., None]) * padded_ive  # (B, I, LI, E)
    uve = F.reshape(uve, (-1,) + uve.shape[2:]) # (B*I, LI, E)
    uve = F.separate(uve, 0)  # B*I x [(LI, E), ...]
    ulens = np.array([len(t) for t in self.inv_examples[0]] * texts.shape[0]) # (I,)
    uve = [seq[:l] for seq, l in zip(uve, ulens)]  # I x [(L1, E), (L2, E), ..]
    # ---------------------------
    # Compute unification predictions
    _, upred = self.predict(uve)  # (B*I, 1)
    upred = F.reshape(upred, (texts.shape[0], self.inv_examples[0].shape[0], 1)) # (B, I, 1)
    upred = F.sum(upred, 1) # (B, 1)
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
    ilabels = self.predictor.inv_examples[1][:, None] # (I, 1)
    for k, t in [('o', labels), ('u', labels), ('ig', ilabels)]:
      report[k + 'loss'] = F.sigmoid_cross_entropy(r[k + 'pred'], t)
      report[k + 'acc'] = F.binary_accuracy(r[k + 'pred'], t)
    # ---------------------------
    # Aux lossess
    vloss = F.mean(r['vmap']) # ()
    report['vloss'] = vloss
    # ---------------------------
    C.report(report, self)
    return self.uniparam*(report['uloss'] + 0.1*report['vloss']) + report['oloss']

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
  # Setup invariant repositories
  idxs = np.random.choice(len(train_data), size=ARGS.invariants, replace=False)
  invariants = train_data[idxs]
  # ---------------------------
  # Setup model
  model = URNN(invariants)
  cmodel = Classifier(model)
  optimiser = C.optimizers.Adam(alpha=ARGS.learning_rate).setup(cmodel)
  train_iter = C.iterators.SerialIterator(train_data, ARGS.batch_size)
  test_iter = C.iterators.SerialIterator(test_data, ARGS.batch_size, repeat=False, shuffle=False)
  updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
  trainer = T.Trainer(updater, (2000, 'iteration'), out='results/urnn_result')
  # ---------------------------
  # Setup debug output
  test_iter.reset()
  debug_texts, debug_labels = converter(next(test_iter), None)
  debug_texts, debug_labels = debug_texts[:4], debug_labels[:4]
  def print_vmap(trainer):
    """Enable unification loss function in model."""
    print_tasks((debug_texts, debug_labels))
    print("INVS:")
    print_tasks(model.inv_examples)
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
    f.write("Inv:\n")
    print_tasks(model.inv_examples, file=f)
    f.write("Out:\n")
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
  train(trainfold, testfold, foldidx)
