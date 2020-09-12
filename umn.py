"""Unification Memory Networks"""
import argparse
import os
import json
import pickle
import uuid
import signal
from collections import OrderedDict
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
parser.add_argument("-r", "--rules", default=3, type=int, help="Number of rules in repository.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-t", "--train_size", default=0, type=int, help="Training size, 0 means use everything.")
parser.add_argument("-w", "--weak", action="store_true", help="Weak supervision setting.")
parser.add_argument("--runc", default=0, type=int, help="Run count of the experiment, for multiple runs.")
ARGS = parser.parse_args()
print("TASK:", ARGS.task)

# Debug
if ARGS.debug:
  # logging.basicConfig(level=logging.DEBUG)
  # C.set_debug(True)
  # import matplotlib
  # matplotlib.use('pdf')
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  import seaborn as sns; sns.set()

EMBED = ARGS.embed
MAX_HIST = 250
REPO_SIZE = ARGS.rules
DROPOUT = 0.1
BABI = 'qa' in ARGS.task
DEEPLOGIC = not BABI
MINUS_INF = -100
STRONG = 0.0 if ARGS.weak else 1.0

# ---------------------------

def load_babi_task(fname):
  """Load task stories from given file name."""
  ss = list()
  with open(fname) as f:
    context = OrderedDict()
    for line in f:
      line = line.strip()
      sid, sl = line.split(' ', 1)
      # Is this a new story?
      sid = int(sid)
      if sid in context:
        context = OrderedDict()
      # Check for question or not
      if '\t' in sl:
        q, a, supps = sl.split('\t')
        idxs = list(context.keys())
        supps = [idxs.index(int(s)) for s in supps.split(' ')]
        cctx = list(context.values())
        # cctx.reverse()
        ss.append({'context': cctx[:MAX_HIST], 'query': q,
                   'answers': [a], 'supps': supps})
      else:
        # Just a statement
        context[sid] = sl
  return ss

def load_deeplogic_task(fname):
  """Load logic programs from given file name."""
  def process_rule(rule):
    """Apply formatting to rule."""
    return rule.replace('.', '').replace('(', ' ( ').replace(')', ' )').replace(':-', ' < ').replace(';', ' ; ').replace(',', ' , ').replace('-', '- ')
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
        ss.append({'context': ctx.copy(), 'query': process_rule(q),
                   'answers': [t], 'supps': supps})
        isnew_ctx = True
      else:
        if isnew_ctx:
          ctx = list()
          isnew_ctx = False
        ctx.append(process_rule(l))
  return ss

loadf = load_babi_task if BABI else load_deeplogic_task
stories = loadf(ARGS.task)
test_stories = loadf(ARGS.task.replace('train', 'test'))
# Print general information
print("EMBED:", EMBED)
print("STRONG:", STRONG)
print("REPO:", REPO_SIZE)
print("TRAIN:", len(stories), "stories")
print("TEST:", len(test_stories), "stories")
print("SAMPLE:", stories[0])

# ---------------------------

# Tokenisation of sentences
def tokenise(text, filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n', split=' '):
  """Lower case naive space based tokeniser."""
  if BABI:
    text = text.lower()
    translate_dict = dict((c, split) for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]

# Word indices
word2idx = {'pad': 0, 'unk': 1}
if DEEPLOGIC:
  word2idx['0'] = 2
  word2idx['1'] = 3

# Encode stories
def encode_story(story):
  """Convert given story into word vector indices."""
  es = dict()
  es['context'] = [np.array([word2idx.setdefault(w, len(word2idx)) for w in tokenise(s)], dtype=np.int32) for s in story['context']]
  es['query'] = np.array([word2idx.setdefault(w, len(word2idx)) for w in tokenise(story['query'])], dtype=np.int32)
  es['answers'] = np.array([word2idx.setdefault(w, len(word2idx)) for w in story['answers']], dtype=np.int32)
  es['supps'] = np.array(story['supps'], dtype=np.int32)
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
else:
  train_enc_stories, val_enc_stories = train_test_split(enc_stories, test_size=0.1)
assert len(train_enc_stories) > REPO_SIZE, "Not enough training stories to generate rules from."
print("TRAIN-VAL:", len(train_enc_stories), '-', len(val_enc_stories))

def decode_story(story):
  """Decode a given story back into words."""
  ds = dict()
  ds['context'] = [[idx2word[widx] for widx in c] for c in story['context']]
  ds['query'] = [idx2word[widx] for widx in story['query']]
  ds['answers'] = [idx2word[widx] for widx in story['answers']]
  ds['supps'] = story['supps']
  return ds

def vectorise_stories(encoded_stories, noise=False):
  """Given a list of encoded stories, vectorise them with padding."""
  # Find maximum length of batch to pad
  max_ctxlen, ctx_maxlen, q_maxlen, a_maxlen, s_maxlen = 0, 0, 0, 0, 0
  for s in encoded_stories:
    max_ctxlen = max(max_ctxlen, len(s['context']))
    c_maxlen = max([len(c) for c in s['context']])
    ctx_maxlen = max(ctx_maxlen, c_maxlen)
    q_maxlen = max(q_maxlen, len(s['query']))
    a_maxlen = max(a_maxlen, len(s['answers']))
    s_maxlen = max(s_maxlen, len(s['supps']))
  # Vectorise stories
  vctx = np.zeros((len(encoded_stories), max_ctxlen, ctx_maxlen), dtype=np.int32) # (B, Cs, C)
  vq = np.zeros((len(encoded_stories), q_maxlen), dtype=np.int32) # (B, Q)
  vas = np.zeros((len(encoded_stories), a_maxlen), dtype=np.int32) # (B, A)
  supps = np.zeros((len(encoded_stories), s_maxlen), dtype=np.int32) # (B, I)
  for i, s in enumerate(encoded_stories):
    vq[i,:len(s['query'])] = s['query']
    vas[i,:len(s['answers'])] = s['answers']
    supps[i] = np.pad(s['supps'], (0, s_maxlen-s['supps'].size), 'constant', constant_values=-1)
    for j, c in enumerate(s['context']):
      vctx[i,j,:len(c)] = c
    # At random convert a symbol to unknown within a story
    if noise and np.random.rand() < DROPOUT:
      words = np.unique(np.concatenate((s['query'], s['answers'], *s['context'])))
      rword = np.random.choice(words)
      vctx[i, vctx[i] == rword] = 1
      vq[i, vq[i] == rword] = 1
      vas[i, vas[i] == rword] = 1
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
          for v in vstory[:-1]]

# ---------------------------

# Utility functions for visualisation
def plot_att_matrix(symbols, att, idxij, outf=None):
  """Plot the unification attention matrix."""
  # Assume one rule, bath_size of 1
  # att.shape == (1, 1, Ps, P, Cs, C)
  isymbols, jsymbols = symbols # (1, Ps, P), (1, Cs, C)
  i, j = idxij
  att = F.swapaxes(att, 3, 4) # (1, 1, Ps, Cs, P, C)
  psyms, csyms, att = isymbols[0,i], jsymbols[0,j], att.array[0,0,i,j] # (P,), (C,), (P, C)
  ylabels = [idx2word[y] for y in psyms if y != 0] # M x ['token', ...]
  xlabels = [idx2word[x] for x in csyms if x != 0] # N x ['token', ...]
  att = att[:len(ylabels), :len(xlabels)] # (M, N)
  # ---------------------------
  # att = np.array([[0.972, 0.011, 0.002, 0.003, 0.012],
                  # [0.003, 0.988, 0.009, 0.   , 0.   ],
                  # [0.08 , 0.465, 0.316, 0.11 , 0.028],
                  # [0.   , 0.   , 0.   , 0.005, 0.995]])
  # ylabels = ['john', 'left', 'the', 'football']
  # xlabels = ['mary', 'got', 'the', 'milk', 'there']
  # ---------------------------
  plt.figure(figsize=(3.2, 2.4))
  ax = sns.heatmap(att, vmin=0, vmax=1, annot=False,
                   linewidths=0.5,
                   cmap='Blues', cbar=False, square=True,
                   xticklabels=xlabels, yticklabels=ylabels,
                   mask=None)
  ax.xaxis.set_ticks_position('top')
  ax.yaxis.set_ticks_position('left')
  plt.xticks(rotation='vertical')
  plt.yticks(rotation='horizontal')
  if outf:
    plt.savefig(outf, bbox_inches='tight')
    with open(outf+'.pkl', 'wb') as f:
      pickle.dump((xlabels, ylabels, att), f)
  else:
    plt.tight_layout()
    plt.show()

# ---------------------------

# Utility functions for neural networks
def sequence_embed(seqs, embed):
  """Embed sequences of integer ids to word vectors."""
  x_len = [len(x) for x in seqs]
  x_section = np.cumsum(x_len[:-1])
  ex = embed(F.concat(seqs, axis=0))
  exs = F.split_axis(ex, x_section, 0)
  return exs

def bow_encode(_, exs):
  """Given sentences compute is bag-of-words representation."""
  # _, (..., S, E)
  return F.sum(exs, -2) # (..., E)

def pos_encode(vxs, exs):
  """Given sentences compute positional encoding."""
  # (..., S), (..., S, E)
  n_words, n_units = exs.shape[-2:] # S, E
  # To avoid 0/0, we use max(length, 1) here.
  length = np.maximum(np.sum((vxs != 0).astype(np.float32), axis=-1), 1) # (...,)
  length = length[..., None, None] # (..., 1, 1)
  k = np.arange(1, n_units + 1, dtype=np.float32) / n_units # (E,)
  i = np.arange(1, n_words + 1, dtype=np.float32)[:, None] # (S, 1)
  coeff = (1 - i / length) - k * (1 - 2.0 * i / length) # (..., S, E)
  enc = coeff * exs # (..., S, E)
  return F.sum(enc, axis=-2) # (..., E)

def seq_rnn_embed(vxs, exs, birnn, return_seqs=False):
  """Embed given sequences using rnn."""
  # vxs.shape == (..., S)
  # exs.shape == (..., S, E)
  assert vxs.shape == exs.shape[:-1], "Sequence embedding dimensions do not match."
  lengths = np.sum(vxs != 0, -1).flatten() # (X,)
  seqs = F.reshape(exs, (-1,)+exs.shape[-2:]) # (X, S, E)
  toembed = [s[..., :l, :] for s, l in zip(F.separate(seqs, 0), lengths) if l != 0] # Y x [(S1, E), (S2, E), ...]
  hs, ys = birnn(None, toembed) # (2, Y, E), Y x [(S1, 2*E), (S2, 2*E), ...]
  if return_seqs:
    ys = F.pad_sequence(ys) # (Y, S, 2*E)
    ys = F.reshape(ys, ys.shape[:-1] + (2, EMBED)) # (Y, S, 2, E)
    ys = F.mean(ys, -2) # (Y, S, E)
    if ys.shape[0] == lengths.size:
      ys = F.reshape(ys, exs.shape) # (..., S, E)
      return ys
    embeds = np.zeros((lengths.size, vxs.shape[-1], EMBED), dtype=np.float32) # (X, S, E)
    idxs = np.nonzero(lengths) # (Y,)
    embeds = F.scatter_add(embeds, idxs, ys) # (X, S, E)
    embeds = F.reshape(embeds, exs.shape) # (..., S, E)
    return embeds # (..., S, E)
  hs = F.mean(hs, 0) # (Y, E)
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

# Memory querying component
class MemAttention(C.Chain):
  """Computes attention over memory components given query."""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.seq_birnn = L.NStepBiGRU(1, EMBED, EMBED, DROPOUT)
      self.att_linear = L.Linear(4*EMBED, EMBED)
      self.att_birnn = L.NStepBiGRU(1, EMBED, EMBED, DROPOUT)
      self.att_score = L.Linear(2*EMBED, 1)
      self.state_linear = L.Linear(4*EMBED, EMBED)

  def seq_embed(self, vxs, exs):
    """Embed a given sequence."""
    # vxs.shape == (..., S)
    # exs.shape == (..., S, E)
    return seq_rnn_embed(vxs, exs, self.seq_birnn)

  def init_state(self, vq, eq):
    """Initialise given state."""
    # vq.shape == (..., S)
    # eq.shape == (..., S, E)
    return self.seq_embed(vq, eq) # (..., E)

  def forward(self, equery, vmemory, ememory, mask, iteration=0):
    """Compute an attention over memory given the query."""
    # equery.shape == (..., E)
    # vmemory.shape == (..., Ms, M)
    # ememory.shape == (..., Ms, E)
    # mask.shape == (..., Ms)
    # Setup memory embedding
    eq = F.repeat(equery[..., None, :], vmemory.shape[-2], -2) # (..., Ms, E)
    # Compute content based attention
    merged = F.concat([eq, ememory, eq*ememory, F.squared_difference(eq, ememory)], -1) # (..., Ms, 4*E)
    inter = self.att_linear(merged, n_batch_axes=len(vmemory.shape)-1) # (..., Ms, E)
    inter = F.tanh(inter) # (..., Ms, E)
    inter = F.dropout(inter, DROPOUT) # (..., Ms, E)
    inter = F.reshape(inter, (-1,) + inter.shape[-2:]) # (X, Ms, E)
    # Split into sentences
    lengths = np.sum(np.any((vmemory != 0), -1), -1).flatten() # (X,)
    mems = [s[..., :l, :] for s, l in zip(F.separate(inter, 0), lengths)] # X x [(M1, E), (M2, E), ...]
    _, bimems = self.att_birnn(None, mems) # X x [(M1, 2*E), (M2, 2*E), ...]
    bimems = F.pad_sequence(bimems) # (X, Ms, 2*E)
    att = self.att_score(bimems, n_batch_axes=2) # (X, Ms, 1)
    att = F.squeeze(att, -1) # (X, Ms)
    att = F.reshape(att, mask.shape) # (..., Ms)
    att += mask * MINUS_INF # (..., Ms)
    return att

  def update_state(self, oldstate, mem_att, vmemory, ememory, iteration=0):
    """Update state given old, attention and new possible states."""
    # oldstate.shape == (..., E)
    # mem_att.shape == (..., Ms)
    # vmemory.shape == (..., Ms, M)
    # ememory.shape == (..., Ms, E)
    ostate = F.repeat(oldstate[..., None, :], vmemory.shape[-2], -2) # (..., Ms, E)
    merged = F.concat([ostate, ememory, ostate*ememory, F.squared_difference(ostate, ememory)], -1) # (..., Ms, 4*E)
    mem_inter = self.state_linear(merged, n_batch_axes=len(merged.shape)-1) # (..., Ms, E)
    mem_inter = F.tanh(mem_inter) # (..., E)
    # (..., Ms) x (..., Ms, E) -> (..., E)
    new_state = F.einsum("...i,...ij->...j", mem_att, mem_inter) # (..., E)
    return new_state

# ---------------------------

# Inference network
class Infer(C.Chain):
  """Takes a story, a set of rules and predicts answers."""
  def __init__(self, rule_stories):
    super().__init__()
    # Setup rule repo
    rvctx, rvq, rva, rsupps = vectorise_stories(rule_stories) # (R, Ls, L), (R, Q), (R, A), (R, I)
    self.add_persistent('rvctx', rvctx)
    self.add_persistent('rvq', rvq)
    self.add_persistent('rva', rva)
    self.add_persistent('rsupps', rsupps)
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED, ignore_label=0)
      # self.rulegen = RuleGen()
      self.vmap_params = C.Parameter(0.0, (rvq.shape[0], len(word2idx)), name='vmap_params') # (R, V)
      self.mematt = MemAttention()
      self.uni_birnn = L.NStepBiGRU(1, EMBED, EMBED, DROPOUT)
      self.uni_linear = L.Linear(EMBED, EMBED, nobias=True)
      self.answer_linear = L.Linear(EMBED, EMBED)
      self.answer_pick = C.Parameter(0.0, (rvq.shape[0],), name='answer_pick') # (R,)
      self.rule_linear = L.Linear(EMBED, 1)
    self.log = None

  @property
  def vrules(self):
    return self.rvctx, self.rvq, self.rva, self.rsupps

  def tolog(self, key, value):
    """Append to log dictionary given key value pair."""
    loglist = self.log.setdefault(key, [])
    loglist.append(value)

  def compute_vmap(self):
    """Compute the variable map for rules."""
    rvctx, rvq, rva, rsupps = self.vrules # (R, Ls, L), (R, Q), (R, A), (R, I)
    rwords = np.reshape(rvctx, (rvctx.shape[0], -1)) # (R, Ls*L)
    rwords = np.concatenate([rvq, rwords], -1) # (R, Q+Ls*L)
    wordrange = np.arange(len(word2idx)) # (V,)
    wordrange[0] = -1 # Null padding is never a variable
    mask = np.vstack([np.isin(wordrange, rws) for rws in rwords]) # (R, V)
    vmap = F.sigmoid(self.vmap_params*10) # (R, V)
    vmap *= mask # (R, V)
    return vmap

  def unification_features(self, vseq, embedded_seq):
    """Compute unification features of an embedded sequence."""
    # vseq.shape = (..., S)
    # embedded_seq.shape = (..., S, E)
    uni_feats = seq_rnn_embed(vseq, embedded_seq, self.uni_birnn, True) # (..., S, E)
    uni_feats = self.uni_linear(uni_feats, n_batch_axes=len(vseq.shape)) # (..., S, E)
    return uni_feats

  def unify_queries(self, rule_q, uni_rule_q, batch_q, uni_batch_q, embedded_batch):
    """Unify given two query sentences."""
    # rule_q.shape = (R, Q)
    # uni_rule_q.shape = (R, Q, E)
    # batch_q.shape = (B, Q')
    # uni_batch_q.shape = (B, Q', E)
    # embedded_batch.shape = (B, Q', E)
    # ---------------------------
    # Setup masks
    mask_rule_q = (rule_q != 0) # (R, Q)
    mask_batch_q = (batch_q == 0) # (B, Q')
    sim_mask = mask_batch_q.astype(np.float32) * MINUS_INF # (B, Q')
    # ---------------------------
    # Calculate similarity of every word to every other word
    # (R, Q, E) x (B, Q', E) -> (B, R, Q, Q')
    raw_sims = F.einsum("rqe,bpe->brqp", uni_rule_q, uni_batch_q) # (B, R, Q, Q')
    # ---------------------------
    # Calculated attended rule query representation
    raw_sims += sim_mask[:, None, None] # (B, R, Q, Q')
    sim_weights = F.softmax(raw_sims, -1) # (B, R, Q, Q')
    sim_weights *= mask_rule_q[..., None] # (B, R, Q, Q')
    # (B, R, Q, Q') x (B, Q', E) -> (B, R, Q, E)
    unifications = F.einsum("brqp,bpe->brqe", sim_weights, embedded_batch)
    return unifications, sim_weights

  def unify_context(self, toprove, uni_toprove, candidates, uni_candidates, embedded_candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (B, R, Ps, P)
    # uni_toprove.shape = (B, R, Ps, P, E)
    # candidates.shape = (B, Cs, C)
    # uni_candidates.shape = (B, Cs, C, E)
    # embedded_candidates.shape = (B, Cs, C, E)
    # ---------------------------
    # Setup masks
    mask_toprove = (toprove != 0) # (B, R, Ps, P)
    mask_candidates = (candidates == 0) # (B, Cs, C)
    sim_mask = mask_candidates.astype(np.float32) * MINUS_INF # (B, Cs, C)
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute similarity between every provable symbol and candidate symbol
    # (..., Ps, P, E) x (B, Cs, C, E)
    raw_sims = F.einsum("brpse,bcde->rpsbcd", uni_toprove, uni_candidates) # (R, Ps, P, B, Cs, C)
    # ---------------------------
    # Calculate attended unified word representations for toprove
    raw_sims += sim_mask # (R, Ps, P, B, Cs, C)
    raw_sims = F.moveaxis(raw_sims, -3, 0) # (B, R, Ps, P, Cs, C)
    sim_weights = F.softmax(raw_sims, -1) # (B, R, Ps, P, Cs, C)
    sim_weights *= mask_toprove[..., None, None] # (B, R, Ps, P, Cs, C)
    # (B, R, Ps, P, Cs, C) x (B, Cs, C, E)
    unifications = F.einsum("brpscd,bcde->brpsce", sim_weights, embedded_candidates) # (B, R, Ps, P, Cs, E)
    return unifications, sim_weights

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    self.log = dict()
    # ---------------------------
    vctx, vq, va, supps = stories # (B, Cs, C), (B, Q), (B, A), (B, I)
    # Embed stories
    ectx = self.embed(vctx) # (B, Cs, C, E)
    eq = self.embed(vq) # (B, Q, E)
    # ---------------------------
    # Prepare rules and variable states
    rvctx, rvq, rva, rsupps = self.vrules # (R, Ls, L), (R, Q), (R, A), (R, I)
    erctx, erq, era = [self.embed(v) for v in self.vrules[:-1]] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    # ---------------------------
    # Compute variable map
    vmap = self.compute_vmap() # (R, V)
    self.tolog('vmap', vmap)
    # ---------------------------
    # Indexing ranges
    nrules_range = np.arange(rvq.shape[0]) # (R,)
    nbatch_range = np.arange(vctx.shape[0]) # (B,)
    # ---------------------------
    # Rule states
    rs = self.mematt.init_state(rvq, erq) # (R, E)
    # Original states
    orig_cs = self.mematt.init_state(vq, eq) # (B, E)
    # ---------------------------
    # Unify query first assuming given query is ground
    uni_erq = self.unification_features(rvq, erq) # (R, Q, E)
    uni_eq = self.unification_features(vq, eq) # (B, Q', E)
    qunis, q_uniatt = self.unify_queries(rvq, uni_erq, vq, uni_eq, eq) # (B, R, Q, E), (B, R, Q, Q')
    self.tolog('q_uniatt', q_uniatt)
    # Unified states
    qvgates = vmap[nrules_range[:, None], rvq] # (R, Q)
    qstate = qvgates[..., None]*qunis + (1-qvgates[..., None])*erq # (B, R, Q, E)
    brvq = np.repeat(rvq[None, ...], qstate.shape[0], 0) # (B, R, Q)
    uni_cs = self.mematt.init_state(brvq, qstate) # (B, R, E)
    # ---
    repeat_orig_cs = F.repeat(orig_cs[:, None], rvctx.shape[0], 1) # (B, R, E)
    uniloss = F.mean(F.squared_difference(uni_cs, repeat_orig_cs), -1) # (B, R)
    self.tolog('uniloss', uniloss)
    # Initialiase and update variable value store
    used_v = F.sum(wordeye[rvq], 1) # (R, V) tells how many scatter additions we did
    unified_v = F.clip(used_v, 0, 1) # (R, V) tells which variables were unified this round
    vvals = (1-unified_v[..., None])*self.embed.W # (R, V, E)
    vvals = F.repeat(vvals[None], vctx.shape[0], 0) # (B, R, V, E)
    vvals = F.scatter_add(vvals, (nbatch_range[:, None, None], nrules_range[:, None], rvq), qunis) # (B, R, V, E)
    norm_v = used_v + (1-unified_v) # (R, V) avoid divide by zero
    vvals /= norm_v[..., None] # (B, R, V, E)
    # Update rule context with variable values
    verctx = vvals[nbatch_range[:, None, None, None], nrules_range[:, None, None], rvctx] # (B, R, Ls, L, E)
    ctxvgates = vmap[nrules_range[:, None, None], rvctx, None] # (R, Ls, L)
    unified_erctx = ctxvgates*verctx + (1-ctxvgates)*erctx # (B, R, Ls, L, E)
    # Update vmap, used variables become constants
    # vmap = vmap*(toswitch*(1-vmap) + (1-toswitch)*vmap) # (R, V)
    vmap = (1-unified_v)*vmap # (R, V)
    self.tolog('vmap', vmap)
    # ---------------------------
    # Setup memory sequence embeddings
    mem_erctx = self.mematt.seq_embed(rvctx, erctx) # (R, Ls, E)
    mem_ectx = self.mematt.seq_embed(vctx, ectx) # (B, Cs, E)
    # ---------------------------
    # Attention masks, and rule repeated story contexts
    rulebodyattmask = np.all(rvctx == 0, -1) # (R, Ls)
    candattmask = np.all(vctx == 0, -1) # (B, Cs)
    repeat_candattmask = np.repeat(candattmask[:, None], rvctx.shape[0], 1) # (B, R, Cs)
    repeat_vctx = np.repeat(vctx[:, None], rvctx.shape[0], 1) # (B, R, Cs, C)
    repeat_mem_ectx = F.repeat(mem_ectx[:, None], rvctx.shape[0], 1) # (B, R, Cs, E)
    repeat_rvctx = np.repeat(rvctx[None], vctx.shape[0], 0) # (B, R, Ls, L)
    uni_ectx = self.unification_features(vctx, ectx) # (B, Cs, C, E)
    # ---------------------------
    # Compute iterative updates on variables
    for t in range(supps.shape[-1]):
      # ---------------------------
      # Compute which body literal to prove using rule state
      raw_body_att = self.mematt(rs, rvctx, mem_erctx, rulebodyattmask, t) # (R, Ls)
      self.tolog('raw_body_att', raw_body_att)
      body_att = F.softmax(raw_body_att, -1) # (R, Ls)
      # Compute unified candidate attention
      raw_uni_cands_att = self.mematt(uni_cs, repeat_vctx, repeat_mem_ectx, repeat_candattmask, t) # (B, R, Cs)
      self.tolog('raw_uni_cands_att', raw_uni_cands_att)
      uni_cands_att = F.softmax(raw_uni_cands_att, -1) # (B, R, Cs)
      # Compute original candidate attention
      raw_orig_cands_att = self.mematt(orig_cs, vctx, mem_ectx, candattmask, t) # (B, Cs)
      self.tolog('raw_orig_cands_att', raw_orig_cands_att)
      orig_cands_att = F.softmax(raw_orig_cands_att, -1) # (B, Cs)
      # ---------------------------
      # Update states for the rule and original
      rs = self.mematt.update_state(rs, body_att, rvctx, mem_erctx, t) # (R, E)
      orig_cs = self.mematt.update_state(orig_cs, orig_cands_att, vctx, mem_ectx, t) # (B, E)
      # ---------------------------
      # Compute attended unification over candidates
      uni_erctx = self.unification_features(repeat_rvctx, unified_erctx) # (B, R, Ls, L, E)
      unis, uni_att = self.unify_context(repeat_rvctx, uni_erctx, vctx, uni_ectx, ectx) # (B, R, Ls, L, Cs, E), (B, R, Ls, L, Cs, C)
      self.tolog('uni_att', uni_att)
      # Using the context attention, select the final unifying sentence
      # (B, R, Cs) x (B, R, Ls, L, Cs, E) -> (B, R, Ls, L, E)
      unis = F.einsum("brc,brlsce->brlse", uni_cands_att, unis) # (B, R, Ls, L, E)
      # ---------------------------
      # Compute unified representation of context and update state
      ctxvgates = vmap[nrules_range[:, None, None], rvctx, None] # (R, Ls, L, 1)
      new_erctx = ctxvgates*unis + (1-ctxvgates)*unified_erctx # (B, R, Ls, L, E)
      mem_uni_erctx = self.mematt.seq_embed(repeat_rvctx, new_erctx) # (B, R, Ls, E)
      repeat_rule_body_att = F.repeat(body_att[None], vctx.shape[0], 0) # (B, R, Ls)
      uni_cs = self.mematt.update_state(uni_cs, repeat_rule_body_att, repeat_rvctx, mem_uni_erctx) # (B, R, E)
      # ---------------------------
      # Update variable value store and propagate values
      used_v = F.sum(wordeye[rvctx], 2) # (R, Ls, V)
      unified_v = F.clip(used_v, 0, 1) # (R, Ls, V)
      vvals = (1-unified_v[..., None])*self.embed.W # (R, Ls, V, E)
      vvals = F.repeat(vvals[None], vctx.shape[0], 0) # (B, R, Ls, V, E)
      vvals = F.scatter_add(vvals, (nbatch_range[:, None, None, None], nrules_range[:, None, None], np.arange(vvals.shape[2])[:, None], rvctx), unis) # (B, R, Ls, V, E)
      norm_v = used_v + (1-unified_v) # (R, Ls, V) avoid divide by zero
      vvals /= norm_v[..., None] # (B, R, Ls, V, E)
      # (R, Ls) x (B, R, Ls, V, E) -> (B, R, V, E)
      vvals = F.einsum("rl,brlve->brve", body_att, vvals) # (B, R, V, E)
      # ---------------------------
      # Compute new context with variable substitution
      verctx = vvals[nbatch_range[:, None, None, None], nrules_range[:, None, None], rvctx] # (B, R, Ls, L, E)
      unified_erctx = ctxvgates*verctx + (1-ctxvgates)*unified_erctx # (B, R, Ls, L, E)
      # ---------------------------
      # Update variable map, used variables become constants
      # (R, Ls) x (R, Ls, V) -> (R, V)
      unified_v = F.einsum("rl,rlv->rv", body_att, unified_v) # (R, V)
      # vmap = vmap*(toswitch*(1-vmap) + (1-toswitch)*vmap) # (R, V)
      vmap = (1-unified_v)*vmap # (R, V)
      self.tolog('vmap', vmap)
      # ---------------------------
      repeat_orig_cs = F.repeat(orig_cs[:, None], rvctx.shape[0], 1) # (B, R, E)
      uniloss = F.mean(F.squared_difference(uni_cs, repeat_orig_cs), -1) # (B, R)
      self.tolog('uniloss', uniloss)
    # ---------------------------
    # Compute answers based on variable and rule scores
    predictions = self.answer_linear(uni_cs, n_batch_axes=2) # (B, R, E)
    vpreds = vvals[nbatch_range[:, None, None], nrules_range[:, None], rva] # (B, R, A, E)
    vpreds = vpreds[:, :, 0] # (B, R, E) no multi-answer support yet
    answer_pick = F.sigmoid(self.answer_pick) # (R,)
    self.tolog('answer_pick', answer_pick)
    predictions = answer_pick[:, None]*vpreds + (1-answer_pick[:, None])*predictions # (B, R, E)
    # ---------------------------
    # Compute rule attentions
    if rvq.shape[0] > 0: # if more than 1 rule
      ratt = self.rule_linear(uni_cs, n_batch_axes=2) # (B, R, 1)
      ratt = F.softmax(F.squeeze(ratt, -1), -1) # (B, R)
    else:
      ratt = np.ones((vctx.shape[0], rvctx.shape[0]), dtype=np.float32) # (B, R)
    self.tolog('ratt', ratt)
    # ---------------------------
    # Compute final word predictions
    # (B, R) x (B, R, E) -> (B, E)
    predictions = F.einsum("br,bre->be", ratt, predictions) # (B, E)
    predictions = predictions @ self.embed.W.T # (B, V)
    # Compute auxilary answers
    rpred = self.answer_linear(rs) # (R, E)
    rpred = rpred @ self.embed.W.T # (R, V)
    self.tolog('rpred', rpred)
    opred = self.answer_linear(orig_cs) # (B, E)
    opred = opred @ self.embed.W.T # (B, V)
    self.tolog('opred', opred)
    return predictions

# ---------------------------

# Wrapper chain for training and predicting
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    self.add_persistent('uniparam', 0.0)
    with self.init_scope():
      self.predictor = predictor

  def forward(self, xin, targets):
    """Compute total loss to train."""
    vctx, vq, va, supps = xin # (B, Cs, C), (B, Q), (B, A), (B, I)
    rvctx, rvq, rva, rsupps = self.predictor.vrules # (R, Ls, L), (R, Q), (R, A), (R, I)
    # ---------------------------
    # Compute main loss
    predictions = self.predictor(xin) # (B, V)
    mainloss = F.softmax_cross_entropy(predictions, targets) # ()
    acc = F.accuracy(predictions, targets) # ()
    # ---------------------------
    # Compute aux losses
    vmaploss = F.sum(self.predictor.log['vmap'][0]) # ()
    ratt = self.predictor.log['ratt'][0] # (B, R)
    uattloss = F.stack(self.predictor.log['raw_uni_cands_att'], 2) # (B, R, I, Cs)
    # (B, R) x (B, R, I, Cs) -> (B, I, Cs)
    uattloss = F.einsum("br,bric->bic", ratt, uattloss) # (B, I, Cs)
    uattloss = F.softmax_cross_entropy(F.reshape(uattloss, (-1, vctx.shape[1])), supps.flatten()) # ()
    # ---
    oattloss = F.stack(self.predictor.log['raw_orig_cands_att'], 1) # (B, I, Cs)
    oattloss = F.softmax_cross_entropy(F.reshape(oattloss, (-1, vctx.shape[1])), supps.flatten()) # ()
    # ---
    battloss = F.stack(self.predictor.log['raw_body_att'], 1) # (R, I, Ls)
    riters = min(rsupps.shape[-1], supps.shape[-1])
    battloss = F.softmax_cross_entropy(F.reshape(battloss[:, :riters], (-1, rvctx.shape[1])), rsupps[:, :riters].flatten()) # ()
    # ---
    rpredloss = F.softmax_cross_entropy(self.predictor.log['rpred'][0], rva[:, 0]) # ()
    opred = self.predictor.log['opred'][0] # (B, V)
    opredloss = F.softmax_cross_entropy(opred, va[:, 0]) # ()
    oacc = F.accuracy(opred, va[:,0]) # ()
    # ---
    uniloss = F.hstack(self.predictor.log['uniloss']) # (I+1,)
    uniloss = F.mean(uniloss) # ()
    # ---
    C.report({'loss': mainloss, 'vmap': vmaploss, 'uatt': uattloss, 'oatt': oattloss, 'batt': battloss, 'rpred': rpredloss, 'opred': opredloss, 'uni': uniloss, 'oacc': oacc, 'acc': acc}, self)
    return self.uniparam*(mainloss + 0.1*vmaploss + STRONG*(uattloss+battloss) + rpredloss + uniloss) + STRONG*oattloss + opredloss # ()

# ---------------------------

# Stories to generate rules from
# Find k top dissimilar questions based on the bag of words
if BABI:
  rule_enc_stories = train_enc_stories # All stories are possible rules
else:
  # We will only induce positive examples to rules following semantics of entailment
  rule_enc_stories = [s for s in train_enc_stories if s['answers'][0] == word2idx['1'] and np.unique(s['query']).size == s['query'].size]
  assert len(rule_enc_stories) >= REPO_SIZE, "Not enough valid rule stories to choose from."
print("VALID RULES:", len(rule_enc_stories))
qs_bow = [np.sum(wordeye[s['query']], 0) for s in rule_enc_stories] # T x (V,)
qs_bow = np.vstack(qs_bow) # (T, V)
qs_bow /= np.linalg.norm(qs_bow, axis=-1, keepdims=True) # (T, V)
qs_sims = qs_bow @ qs_bow.T # (T, T)
# Start with first story
rule_idxs = [0]
while len(rule_idxs) < REPO_SIZE:
  dsims = qs_sims[rule_idxs].T # (T, R)
  dsims = np.mean(dsims, -1) # (T,)
  sidxs = np.argsort(dsims) # (T,)
  # Select a new dissimilar story
  sargmax = np.argmax(np.isin(sidxs, rule_idxs, invert=True))
  sidx = sidxs[sargmax]
  rule_idxs.append(sidx)
rule_repo = [rule_enc_stories[i] for i in rule_idxs] # R x
print("RULE REPO:", rule_repo)

# ---------------------------

# Setup model
model = Infer(rule_repo)
cmodel = Classifier(model)

optimiser = C.optimizers.Adam().setup(cmodel)
if BABI:
  optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))
# optimiser.add_hook(C.optimizer_hooks.GradientClipping(40))

train_iter = C.iterators.SerialIterator(train_enc_stories, 64)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas, supps = vectorise_stories(batch_stories, noise=False) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas, supps), vas[:, 0] # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (4000, 'iteration'), out='results/umn_result')
fname = ARGS.name or ('debug' if ARGS.debug else '') or str(uuid.uuid4())

# Save run parameters
params = {
  'task': ARGS.task,
  'name': fname,
  'rules': REPO_SIZE,
  'weak': ARGS.weak,
  'embed': EMBED,
  'train_size': ARGS.train_size,
  'runc': ARGS.runc
}

with open(trainer.out + '/' + fname + '_params.json', 'w') as f:
  json.dump(params, f)
  print("Saved run parameters.")

# Trainer extensions
def enable_unification(trainer):
  """Enable unification loss function in model."""
  trainer.updater.get_optimizer('main').target.uniparam = 1.0
trainer.extend(enable_unification, trigger=(200, 'iteration'))

def log_vmap(trainer):
  """Log inner properties to file."""
  pmodel = trainer.updater.get_optimizer('main').target.predictor
  vmaplog = pmodel.log['vmap'][0] # (V,)
  logpath = os.path.join(trainer.out, fname + '_vmap.jsonl')
  with open(logpath, 'a') as f:
    if trainer.updater.epoch == 1:
      # Log the rule as well
      f.write("---ENC RULE REPO---\n")
      f.write(str(pmodel.vrules) + '\n')
      f.write("-------------------\n")
    f.write(str(trainer.updater.epoch) + "," + json.dumps(vmaplog.array.tolist()) + '\n')
# trainer.extend(log_vmap, trigger=(10, 'iteration'))

# Validation extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1), name='val', trigger=(10, 'iteration'))
test_iter = C.iterators.SerialIterator(test_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(test_iter, cmodel, converter=converter, device=-1), name='test', trigger=(10, 'iteration'))
# trainer.extend(T.extensions.snapshot(filename=fname+'_best.npz'), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
trainer.extend(T.extensions.snapshot(filename=fname+'_latest.npz'), trigger=(100, 'iteration'))
trainer.extend(T.extensions.LogReport(log_name=fname+'_log.json', trigger=(10, 'iteration')))
# trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=fname+'_log.json'))
trainer.extend(T.extensions.FailOnNonNumber())
report_keys = ['loss', 'vmap', 'uatt', 'oatt', 'batt', 'rpred', 'opred', 'uni', 'oacc', 'acc']
trainer.extend(T.extensions.PrintReport(['iteration'] + ['main/'+s for s in report_keys] + [p+'/main/'+s for p in ('val', 'test') for s in ('loss', 'acc')] + ['elapsed_time']))
# trainer.extend(T.extensions.ProgressBar(update_interval=10))
# trainer.extend(T.extensions.PlotReport(['main/loss', 'validation/main/loss'], 'iteration', marker=None, file_name=fname+'_loss.pdf'))
# trainer.extend(T.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'iteration', marker=None, file_name=fname+'_acc.pdf'))

# Setup training pausing
trainer_statef = trainer.out + '/' + fname + '_latest.npz'
def interrupt(signum, frame):
  """Save and interrupt training."""
  print("Getting interrupted.")
  raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)

# Check previously saved trainer
if os.path.isfile(trainer_statef):
  model.rvctx, model.rvq, model.rva, model.rvs = None, None, None, None
  C.serializers.load_npz(trainer_statef, trainer)
  print("Loaded trainer state from:", trainer_statef)
  print("UNI:", trainer.updater.get_optimizer('main').target.uniparam)
  print("RULES:", trainer.updater.get_optimizer('main').target.predictor.vrules)

# Hit the train button
try:
  trainer.run()
except KeyboardInterrupt:
  pass

# Save latest state
C.serializers.save_npz(trainer_statef, trainer)
print("Saved trainer file:", trainer_statef)

# Collect final rules for inspection
debug_enc_stories = vectorise_stories(test_enc_stories[:10]) # ...
answer = model(debug_enc_stories)[0].array
to_pickle = {
  'debug_enc_stories': debug_enc_stories,
  'debug_stories': decode_vector_stories(debug_enc_stories),
  'answer': answer,
  'vrules': model.vrules,
  'rules': decode_vector_stories(model.vrules),
  'model_log': model.log
}
with open(trainer.out + '/' + fname + '_out.pickle', 'wb') as f:
  pickle.dump(to_pickle, f)
  print("Saved output pickle file.")

# Extra inspection if we are debugging
if ARGS.debug:
  for test_story in test_enc_stories:
    test_story_in, test_story_answer = converter([test_story], None)
    with C.using_config('train', False):
      answer = model(test_story_in)
    prediction = np.argmax(answer.array)
    expected = test_story_answer[0]
    if prediction != expected:
      print(decode_story(test_story))
      print(test_story_in)
      print(f"Expected {expected} '{idx2word[expected]}' got {prediction} '{idx2word[prediction]}'.")
      import ipdb; ipdb.set_trace()
      with C.using_config('train', False):
        answer = model(test_story_in)
  print(decode_story(test_story))
  print(f"Expected {expected} '{idx2word[expected]}' got {prediction} '{idx2word[prediction]}'.")
  print(model.log)
  import ipdb; ipdb.set_trace()
  with C.using_config('train', False):
    answer = model(converter([test_story], None)[0])
  # Plot Embeddings
  pca = PCA(2)
  embds = pca.fit_transform(model.embed.W.array)
  print("PCA VAR:", pca.explained_variance_ratio_)
  plt.scatter(embds[:, 0], embds[:, 1])
  for idx, word in idx2word.items():
    plt.annotate(word, xy=(embds[idx,0], embds[idx,1]), xytext=(10, 10), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  plt.show()
