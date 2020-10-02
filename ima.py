"""Iterative Memory Attention"""
import argparse
import os
import json
import pickle
import uuid
import signal
import numpy as np
from sklearn.model_selection import train_test_split
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T


# Disable scientific printing
np.set_printoptions(suppress=True, precision=3, linewidth=180)
# pylint: disable=line-too-long

# Arguments
parser = argparse.ArgumentParser(description="Run UMN on given tasks.")
parser.add_argument("task", help="File that contains task train.")
parser.add_argument("--name", help="Name prefix for saving files etc.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-t", "--train_size", default=0, type=int, help="Training size, 0 means use everything.")
parser.add_argument("-w", "--weak", action="store_true", help="Weak supervision setting.")
parser.add_argument("--runc", default=0, type=int, help="Run count of the experiment, for multiple runs.")
ARGS = parser.parse_args()
print("TASK:", ARGS.task)

EMBED = ARGS.embed
DEEPLOGIC = True
DROPOUT = 0.1
MINUS_INF = -100
STRONG = 0.0 if ARGS.weak else 1.0

# ---------------------------

def load_deeplogic_task(fname):
  """Load logic programs from given file name."""
  def process_rule(rule):
    """Apply formatting to rule."""
    return rule.replace(':-', '.').replace(';', '.').split('.')[:-1]
  ss = list()
  with open(fname) as f:
    ctx, isnew_ctx = list(), False
    for l in f:
      l = l.strip()
      if l and l[0] == '?':
        _, q, t, supps = l.split(' ')
        supps = [int(s) for s in supps.split(',')]
        if -1 in supps:
          # Ensure partial supervision
          assert len(set(supps[supps.index(-1):])) == 1, "Backtracking supervision in deeplogic task."
        ss.append({'context': ctx.copy(), 'query': process_rule(q)[0],
                   'answers': 1 if t == '1' else 0, 'supps': supps})
        isnew_ctx = True
      else:
        if isnew_ctx:
          ctx = list()
          isnew_ctx = False
        ctx.append(process_rule(l))
  return ss

stories = load_deeplogic_task(ARGS.task)
test_stories = load_deeplogic_task(ARGS.task.replace('train', 'test'))

# ----------
# Print general information
print("EMBED:", EMBED)
print("STRONG:", STRONG)
print("TRAIN:", len(stories), "stories")
print("TEST:", len(test_stories), "stories")
print("SAMPLE:", stories[0])

# ---------------------------

# Tokenisation of predicates
def tokenise(text):
  """Character based tokeniser."""
  return list(text) # p(a) ['p', '(', 'a', ')']

# Word indices
word2idx = {'pad': 0, 'unk': 1}

# Encode stories
def encode_story(story):
  """Convert given story into word vector indices."""
  es = dict()
  enc_ctx = [[[word2idx.setdefault(c, len(word2idx)) for c in tokenise(pred)]
              for pred in rule]
             for rule in story['context']]
  es['context'] = enc_ctx
  es['query'] = [word2idx.setdefault(w, len(word2idx)) for w in tokenise(story['query'])]
  es['answers'] = story['answers']
  es['supps'] = story['supps']
  return es
enc_stories = list(map(encode_story, stories))
print("TRAIN VOCAB:", len(word2idx))
test_enc_stories = list(map(encode_story, test_stories))
print("TEST VOCAB:", len(word2idx))
print("ENC SAMPLE:", enc_stories[0])
idx2word = {v:k for k, v in word2idx.items()}
wordeye = np.eye(len(word2idx), dtype=np.float32)

# Prepare training validation sets
if ARGS.train_size != 0:
  assert ARGS.train_size < len(enc_stories), "Not enough examples for training size."
  tratio = (len(enc_stories)-ARGS.train_size) / len(enc_stories)
  train_enc_stories, val_enc_stories = train_test_split(enc_stories, test_size=tratio)
  while len(train_enc_stories) < 900:
    train_enc_stories.append(np.random.choice(train_enc_stories))
else:
  train_enc_stories, val_enc_stories = train_test_split(enc_stories, test_size=0.1)
print("TRAIN-VAL:", len(train_enc_stories), '-', len(val_enc_stories))

def decode_story(story):
  """Decode a given story back into words."""
  ds = dict()
  ds['context'] = [[''.join([idx2word[cid] for cid in pred]) for pred in rule] for rule in story['context']]
  ds['query'] = ''.join([idx2word[widx] for widx in story['query']])
  ds['answers'] = story['answers']
  ds['supps'] = story['supps']
  return ds

def vectorise_stories(encoded_stories):
  """Given a list of encoded stories, vectorise them with padding."""
  # Vectorise stories
  vctx = np.zeros((len(encoded_stories),
                    max([len(s['context']) for s in encoded_stories]),
                    max([len(rule) for s in encoded_stories for rule in s['context']]),
                    max([len(pred) for s in encoded_stories for rule in s['context'] for pred in rule])),
                   dtype=np.int32) # (B, R, P, C)
  vq = F.pad_sequence([np.array(s['query'], dtype=np.int32) for s in encoded_stories]).array # (B, Q)
  vas = np.array([s['answers'] for s in encoded_stories], dtype=np.int32) # (B,)
  supps = F.pad_sequence([np.array(s['supps'], dtype=np.int32) for s in encoded_stories], padding=-1).array # (B, I)
  for i, s in enumerate(encoded_stories):
    for j, rule in enumerate(s['context']):
      for k, pred in enumerate(rule):
        vctx[i,j,k,:len(pred)] = np.array(pred, dtype=np.int32)
    if DEEPLOGIC:
      perm = np.random.permutation(len(s['context']))
      vctx[i,:len(s['context'])] = vctx[i,perm]
      for j, supp in enumerate(supps[i]):
        if supp != -1:
          supps[i,j] = np.argmax(perm==supp)
  return vctx, vq, vas, supps

def decode_vector_stories(vstory):
  """Decode a given vector of stories."""
  return [np.array([idx2word[i] for i in v.flatten()]).reshape(v.shape)
          for v in vstory[:2]] + list(vstory[2:])

# ---------------------------

# Utility functions for neural networks
def seq_rnn_embed(vxs, exs, rnn_layer, initial_state=None, reverse=False):
  """Embed given sequences using rnn."""
  # vxs.shape == (..., S)
  # exs.shape == (..., S, E)
  # initial_state == (..., E)
  assert vxs.shape == exs.shape[:-1], "Sequence embedding dimensions do not match."
  lengths = np.sum(vxs != 0, -1).flatten() # (X,)
  seqs = F.reshape(exs, (-1,)+exs.shape[-2:]) # (X, S, E)
  if reverse:
    toembed = [F.flip(s[..., :l, :], -2) for s, l in zip(F.separate(seqs, 0), lengths) if l != 0] # Y x [(S1, E), (S2, E), ...]
  else:
    toembed = [s[..., :l, :] for s, l in zip(F.separate(seqs, 0), lengths) if l != 0] # Y x [(S1, E), (S2, E), ...]
  if initial_state is not None:
    initial_state = F.reshape(initial_state, (-1, EMBED)) # (X, E)
    initial_state = initial_state[None, np.flatnonzero(lengths)] # (1, Y, E)
  hs, ys = rnn_layer(initial_state, toembed) # (1, Y, E), Y x [(S1, 2*E), (S2, 2*E), ...]
  hs = hs[0] # (Y, E)
  if hs.shape[0] == lengths.size:
    hs = F.reshape(hs, vxs.shape[:-1] + (EMBED,)) # (..., E)
    return hs
  # Add zero values back to match original shape
  embeds = np.zeros((lengths.size, EMBED), dtype=np.float32) # (X, E)
  idxs = np.nonzero(lengths) # (Y,)
  embeds = F.scatter_add(embeds, idxs, hs) # (X, E)
  embeds = F.reshape(embeds, vxs.shape[:-1] + (EMBED,)) # (..., E)
  return embeds # (..., E)

# ---------------------------

# Iterative Memory Attention
class IterativeMemoryAttention(C.Chain):
  """Takes a logic program, query and predicts entailment."""
  def __init__(self):
    super().__init__()
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED, ignore_label=0)
      self.pred_rnn = L.NStepGRU(1, EMBED, EMBED, DROPOUT)
      self.att_dense1 = L.Linear(5*EMBED, EMBED//2)
      self.att_dense2 = L.Linear(EMBED//2, 1)
      self.unifier = L.NStepGRU(1, EMBED, EMBED, DROPOUT)
      self.out_linear = L.Linear(EMBED, 1)
    self.log = None

  def tolog(self, key, value):
    """Append to log dictionary given key value pair."""
    loglist = self.log.setdefault(key, [])
    loglist.append(value)

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    self.log = dict()
    # ---------------------------
    vctx, vq, va, supps = stories # (B, R, P, C), (B, Q), (B,), (B, I)
    # Embed stories
    # ectx = F.embed_id(vctx, wordeye, ignore_label=0) # (B, R, P, C, V)
    # eq = F.embed_id(vq, wordeye, ignore_label=0) # (B, Q, V)
    ectx = self.embed(vctx) # (B, R, P, C, V)
    eq = self.embed(vq) # (B, Q, V)
    # ---------------------------
    # Embed predicates
    embedded_preds = seq_rnn_embed(vctx, ectx, self.pred_rnn, reverse=True) # (B, R, P, E)
    vector_preds = vctx[..., 0] # (B, R, P) first character to check if pred is empty
    embedded_query = seq_rnn_embed(vq, eq, self.pred_rnn, reverse=True) # (B, E)
    embedded_rules = embedded_preds[:, :, 0] # (B, R, E) head of rule
    # ---------------------------
    # Perform iterative updates
    state = embedded_query # (B, E)
    repeated_query = F.repeat(embedded_query[:, None], vctx.shape[1], 1) # (B, R, E)
    rule_mask = np.all(vctx == 0, (2, 3)) # (B, R)
    for _ in range(supps.shape[-1]):
      # Compute attention over memory
      repeated_state = F.repeat(state[:, None], vctx.shape[1], 1) # (B, R, E)
      combined = F.concat([repeated_state,
                           embedded_rules,
                           repeated_query,
                           F.squared_difference(repeated_state, embedded_rules),
                           embedded_rules * repeated_state], -1) # (B, R, 5*E)
      att = F.tanh(self.att_dense1(combined, n_batch_axes=2)) # (B, R, E//2)
      att = self.att_dense2(att, n_batch_axes=2) # (B, R, 1)
      att = F.squeeze(att, -1) # (B, R)
      att += rule_mask * MINUS_INF # (B, R)
      self.tolog('raw_att', att)
      att = F.softmax(att) # (B, R)
      self.tolog('att', att)
      # Iterate state
      new_states = seq_rnn_embed(vector_preds, embedded_preds, self.unifier, initial_state=repeated_state) # (B, R, E)
      # Update state
      # (B, R) x (B, R, E) -> (B, E)
      state = F.einsum('br,bre->be', att, new_states) # (B, E)
    return self.out_linear(state)[:, 0] # (B,)

# ---------------------------

# Wrapper chain for training and predicting
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    with self.init_scope():
      self.predictor = predictor

  def forward(self, xin, targets):
    """Compute total loss to train."""
    vctx, vq, va, supps = xin # (B, R, P, C), (B, Q), (B,), (B, I)
    # ---------------------------
    # Compute main loss
    predictions = self.predictor(xin) # (B,)
    mainloss = F.sigmoid_cross_entropy(predictions, targets) # ()
    acc = F.binary_accuracy(predictions, targets) # ()
    # ---------------------------
    # Compute aux losses
    oattloss = F.stack(self.predictor.log['raw_att'], 1) # (B, I, R)
    oattloss = F.reshape(oattloss, (-1, vctx.shape[1])) # (B*I, R)
    oattloss = F.softmax_cross_entropy(oattloss, supps.flatten()) # ()
    # ---
    C.report({'loss': mainloss, 'oatt': oattloss, 'acc': acc}, self)
    return mainloss + STRONG*oattloss # ()

# ---------------------------

# Setup model
model = IterativeMemoryAttention()
cmodel = Classifier(model)
optimiser = C.optimizers.Adam().setup(cmodel)
train_iter = C.iterators.SerialIterator(train_enc_stories, 64)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas, supps = vectorise_stories(batch_stories) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas, supps), vas # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (300, 'epoch'), out='results/ima_result')
fname = ARGS.name or ('debug' if ARGS.debug else '') or str(uuid.uuid4())

# Save run parameters
params = {
  'task': ARGS.task,
  'name': fname,
  'weak': ARGS.weak,
  'embed': EMBED,
  'train_size': ARGS.train_size,
  'runc': ARGS.runc
}

with open(trainer.out + '/' + fname + '_params.json', 'w') as f:
  json.dump(params, f)
  print("Saved run parameters.")

# Trainer extensions
# Validation extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1), name='val', trigger=(1, 'epoch'))
test_iter = C.iterators.SerialIterator(test_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(test_iter, cmodel, converter=converter, device=-1), name='test', trigger=(1, 'epoch'))
trainer.extend(T.extensions.snapshot(filename=fname+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(log_name=fname+'_log.json', trigger=(1, 'epoch')))
trainer.extend(T.extensions.FailOnNonNumber())
report_keys = ['loss', 'oatt', 'acc']
trainer.extend(T.extensions.PrintReport(['epoch'] + ['main/'+s for s in report_keys] + [p+'/main/'+s for p in ('val', 'test') for s in report_keys] + ['elapsed_time']))

# Setup training pausing
trainer_statef = trainer.out + '/' + fname + '_latest.npz'
def interrupt(signum, frame):
  """Save and interrupt training."""
  print("Getting interrupted.")
  raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)

# Check previously saved trainer
if os.path.isfile(trainer_statef):
  C.serializers.load_npz(trainer_statef, trainer)
  print("Loaded trainer state from:", trainer_statef)

# Hit the train button
try:
  trainer.run()
except KeyboardInterrupt:
  pass

# Collect final logs for inspection
debug_enc_stories = vectorise_stories(test_enc_stories[:10]) # ...
answer = model(debug_enc_stories).array # (B,)
to_pickle = {
  'debug_enc_stories': debug_enc_stories,
  'debug_stories': decode_vector_stories(debug_enc_stories),
  'answer': answer,
  'model_log': model.log
}
with open(trainer.out + '/' + fname + '_out.pickle', 'wb') as f:
  pickle.dump(to_pickle, f)
  print("Saved output pickle file.")
