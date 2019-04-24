"""bAbI run on neurolog."""
import argparse
import logging
import os
import json
import signal
import time
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T
# import matplotlib
# matplotlib.use('pdf')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Disable scientific printing
np.set_printoptions(suppress=True, precision=3, linewidth=180)

# Arguments
parser = argparse.ArgumentParser(description="Run NeuroLog on bAbI tasks.")
parser.add_argument("task", help="File that contains task train.")
parser.add_argument("name", help="Name prefix for saving files etc.")
parser.add_argument("-r", "--rules", default=3, type=int, help="Number of rules in repository.")
parser.add_argument("-e", "--embed", default=32, type=int, help="Embedding size.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
parser.add_argument("-t", "--tsize", default=0, type=int, help="Training size, 0 means use everything.")
parser.add_argument("-s", "--strong", default=1.0, type=float, help="Strong supervision ratio.")
parser.add_argument("-i", "--iter", default=0, type=int, help="Number of iterations, default 0 for minimum required.")
ARGS = parser.parse_args()
print("TASK:", ARGS.task)

# Debug
# if ARGS.debug:
  # logging.basicConfig(level=logging.DEBUG)
  # C.set_debug(True)

EMBED = ARGS.embed
MAX_HIST = 250
REPO_SIZE = ARGS.rules
ITERATIONS = ARGS.iter
DROPOUT = 0.1
BABI = 'qa' in ARGS.task
DEEPLOGIC = not BABI
MINUS_INF = -100

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
        q, a, supps= sl.split('\t')
        idxs = list(context.keys())
        supps = [idxs.index(int(s)) for s in supps.split(' ')]
        cctx = list(context.values())
        # cctx.reverse()
        ss.append({'context': cctx[:MAX_HIST], 'query': q,
                   'answers': a.split(','), 'supps': supps})
      else:
        # Just a statement
        context[sid] = sl
  return ss

def load_deeplogic_task(fname):
  """Load logic programs from given file name."""
  def process_rule(r):
    """Apply formatting to rule."""
    return r.replace('.', '').replace('(', ' ( ').replace(')', ' )').replace(':-', ' < ').replace(';', ' ; ').replace(',', ' , ')
  ss = list()
  with open(fname) as f:
    ctx, isnew_ctx = list(), False
    for l in f:
      l = l.strip()
      if l and l[0] == '?':
        _, q, t = l.split()
        ss.append({'context': ctx.copy(), 'query': process_rule(q),
                   'answers': [t], 'supps': np.random.choice(len(ctx), size=ITERATIONS or 4)})
        isnew_ctx = True
      else:
        if isnew_ctx:
          ctx = list()
          isnew_ctx = False
        ctx.append(process_rule(l))
  return ss

if BABI:
  # We have a bAbI task
  stories = load_babi_task(ARGS.task)
  max_supps = max([len(s['supps']) for s in stories])
  if not ITERATIONS:
    ITERATIONS = max_supps
  else:
    assert max_supps <= ITERATIONS, "Not enough iterations for supporting facts."
  test_stories = load_babi_task(ARGS.task.replace('train', 'test'))
elif DEEPLOGIC:
  # We have a deeplogic task
  stories = load_deeplogic_task(ARGS.task)
  ITERATIONS = ITERATIONS or 4
  test_stories = load_deeplogic_task(ARGS.task.replace('train', 'test'))
# Print general information
print("EMBED:", EMBED)
print("ITER:", ITERATIONS)
print("STRONG:", ARGS.strong)
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
if ARGS.tsize != 0:
  assert ARGS.tsize < len(enc_stories), "Not enough examples for training size."
  tratio = (len(enc_stories)-ARGS.tsize) / 1000
  train_enc_stories, val_enc_stories = train_test_split(enc_stories, test_size=tratio)
  while len(train_enc_stories) < 900:
    train_enc_stories.append(np.random.choice(train_enc_stories))
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
  max_ctxlen, ctx_maxlen, q_maxlen, a_maxlen = 0, 0, 0, 0
  for s in encoded_stories:
    max_ctxlen = max(max_ctxlen, len(s['context']))
    c_maxlen = max([len(c) for c in s['context']])
    ctx_maxlen = max(ctx_maxlen, c_maxlen)
    q_maxlen = max(q_maxlen, len(s['query']))
    a_maxlen = max(a_maxlen, len(s['answers']))
  # Vectorise stories
  vctx = np.zeros((len(encoded_stories), max_ctxlen, ctx_maxlen), dtype=np.int32) # (B, Cs, C)
  vq = np.zeros((len(encoded_stories), q_maxlen), dtype=np.int32) # (B, Q)
  vas = np.zeros((len(encoded_stories), a_maxlen), dtype=np.int32) # (B, A)
  supps = np.zeros((len(encoded_stories), ITERATIONS), dtype=np.int32) # (B, I)
  for i, s in enumerate(encoded_stories):
    vq[i,:len(s['query'])] = s['query']
    vas[i,:len(s['answers'])] = s['answers']
    supps[i] = np.pad(s['supps'], (0, ITERATIONS-s['supps'].size), 'edge')
    if DEEPLOGIC:
      np.random.shuffle(s['context'])
    for j, c in enumerate(s['context']):
      vctx[i,j,:len(c)] = c
    # At random convert a symbol to unknown within a story
    if noise and np.random.rand() < DROPOUT:
      words = np.unique(np.concatenate((s['query'], s['answers'], *s['context'])))
      rword = np.random.choice(words)
      vctx[i, vctx[i] == rword] = 1
      vq[i, vq[i] == rword] = 1
      vas[i, vas[i] == rword] = 1
  return vctx, vq, vas, supps

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

def contextual_convolve(backend, convolution, vxs, exs):
  """Given vectorised and encoded sentences convolve over last dimension."""
  # (B, ..., S), (B, ..., S, E)
  mask = (vxs != 0.0) # (B, ..., S)
  toconvolve = exs # Assuming we have (B, S), (B, S, E)
  if len(vxs.shape) > 2:
    # We need to pad between sentences
    padding = backend.zeros(exs.shape[:-2]+(1,exs.shape[-1]), dtype=exs.dtype) # (B, ..., 1, E)
    padded = F.concat([exs, padding], -2) # (B, ..., S+1, E)
    toconvolve = F.reshape(padded, (exs.shape[0], -1, exs.shape[-1])) # (B, *S+1, E)
  permuted = F.transpose(toconvolve, (0, 2, 1)) # (B, E, S)
  contextual = convolution(permuted) # (B, E, S)
  contextual = F.transpose(contextual, (0, 2, 1)) # (B, S, E)
  if len(vxs.shape) > 2:
    contextual = F.reshape(contextual, padded.shape) # (B, ..., S+1, E)
    contextual = contextual[..., :-1, :] # (B, ..., S, E)
  contextual *= mask[..., None] # (B, ..., S, E)
  return contextual

def seq_rnn_embed(vxs, exs, birnn, return_seqs=False, init_state=None):
  """Embed given sequences using rnn."""
  # vxs.shape == (..., S)
  # exs.shape == (..., S, E)
  lengths = np.sum(vxs != 0, -1).flatten() # (X,)
  seqs = F.reshape(exs, (-1,)+exs.shape[-2:]) # (X, S, E)
  toembed = [s[..., :l, :] for s, l in zip(F.separate(seqs, 0), lengths) if l != 0] # Y x [(S1, E), (S2, E), ...]
  hs, ys = birnn(init_state, toembed) # (2, Y, E), Y x [(S1, 2*E), (S2, 2*E), ...]
  if return_seqs:
    ys = F.pad_sequence(ys) # (Y, S, 2*E)
    ys = F.reshape(ys, ys.shape[:-1] + (2, EMBED)) # (Y, S, 2, E)
    ys = F.mean(ys, -2) # (Y, S, E)
    embeds = np.zeros((lengths.size, vxs.shape[-1], EMBED), dtype=np.float32) # (X, S, E)
    idxs = np.nonzero(lengths) # (Y,)
    embeds = F.scatter_add(embeds, idxs, ys) # (X, S, E)
    embeds = F.reshape(embeds, exs.shape) # (..., S, E)
    return embeds # (..., S, E)
  else:
    hs = F.mean(hs, 0) # (Y, E)
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
    # return pos_encode(vxs, exs)
    return seq_rnn_embed(vxs, exs, self.seq_birnn)

  def init_state(self, vq, eq):
    """Initialise given state."""
    # vq.shape == (..., S)
    # eq.shape == (..., S, E)
    s = self.seq_embed(vq, eq) # (..., E)
    s = F.dropout(s, DROPOUT) # (..., E)
    return s # (..., E)

  def forward(self, equery, vmemory, ememory, mask=None, iteration=0):
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
    # Split into sentences
    lengths = np.sum(np.any((vmemory != 0), -1), -1) # (...,)
    mems = [s[..., :l, :] for s, l in zip(F.separate(inter, 0), lengths)] # B x [(M1, E), (M2, E), ...]
    _, bimems = self.att_birnn(None, mems) # B x [(M1, 2*E), (M2, 2*E), ...]
    bimems = F.pad_sequence(bimems) # (..., Ms, 2*E)
    att = self.att_score(bimems, n_batch_axes=len(vmemory.shape)-1) # (..., Ms, 1)
    att = F.squeeze(att, -1) # (..., Ms)
    if mask is not None:
      att += mask * MINUS_INF # (..., Ms)
    return att

  def update_state(self, oldstate, mem_att, vmemory, ememory, iteration=0):
    """Update state given old, attention and new possible states."""
    # oldstate.shape == (..., E)
    # mem_att.shape == (..., Ms)
    # vmemory.shape == (..., Ms, M)
    # ememory.shape == (..., Ms, E)
    # (..., Ms) x (..., Ms, E) -> (..., E)
    mem_slot = F.einsum("...i,...ij->...j", mem_att, ememory) # (..., E)
    merged = F.concat([oldstate, mem_slot, oldstate*mem_slot, F.squared_difference(oldstate, mem_slot)], -1) # (..., 3*E)
    new_state = self.state_linear(merged, n_batch_axes=len(merged.shape)-1) # (..., E)
    new_state = F.tanh(new_state) # (..., E)
    new_state = F.dropout(new_state, DROPOUT) # (..., E)
    return new_state

# ---------------------------

# End-to-End Memory Network
class MemN2N(C.Chain):
  """Compute iterations over memory using end-to-end mem approach."""
  def __init__(self):
    super().__init__()
    initW = C.initializers.Normal(0.1)
    with self.init_scope():
      self.embedAC = C.ChainList(*[L.EmbedID(len(word2idx), EMBED, initialW=initW) for _ in range(ITERATIONS+1)])
      self.temporal = C.ChainList(*[L.EmbedID(MAX_HIST, EMBED, initialW=initW) for _ in range(ITERATIONS+1)])

  def seq_embed(self, vxs, iteration=0):
    """Embed a given sequence."""
    # vxs.shape == (..., S)
    exs = self.embedAC[iteration](vxs) # (..., S, E)
    mask = (vxs != 0)[..., None] # (..., S, 1)
    exs *= mask # (..., S, E)
    return pos_encode(vxs, exs) # (..., E)

  def init_state(self, vq, vctx):
    """Initialise given state."""
    # vq.shape == (..., S)
    # vctx.shape == (..., Cs, C)
    s = self.seq_embed(vq) # (..., E)
    return s # (..., E)

  def forward(self, equery, vmemory, mask=None, iteration=0):
    """Compute an attention over memory given the query."""
    # equery.shape == (..., E)
    # vmemory.shape == (..., Ms, M)
    # mask.shape == (..., Ms)
    ememory = self.seq_embed(vmemory, iteration) # (..., Ms, E)
    tidxs = self.xp.arange(vmemory.shape[-2], dtype=self.xp.int32) # (Ms,)
    temps = self.temporal[iteration](tidxs) # (Ms, E)
    em = ememory + temps # (..., Ms, E)
    att = F.einsum("...j,...ij->...i", equery, em) # (..., Ms)
    if mask is not None:
      att += mask * MINUS_INF # (..., Ms)
    return att

  def update_state(self, oldstate, mem_att, vmemory, iteration=0):
    """Update state given old, attention and new possible states."""
    # oldstate.shape == (..., E)
    # mem_att.shape == (..., Ms)
    # vmemory.shape == (..., Ms, M)
    # Setup memory output embedding
    ememory = self.seq_embed(vmemory, iteration+1) # (..., Ms, E)
    tidxs = self.xp.arange(vmemory.shape[-2], dtype=self.xp.int32) # (Ms,)
    temps = self.temporal[iteration+1](tidxs) # (Ms, E)
    em = ememory + temps # (..., Ms, E)
    select_mem = F.einsum("...i,...ij->...j", mem_att, em) # (..., E)
    return select_mem

# ---------------------------
class Embed:
  def __call__(self, x):
    return F.embed_id(x, wordeye, ignore_label=0)

# Inference network
class Infer(C.Chain):
  """Takes a story, a set of rules and predicts answers."""
  def __init__(self, rule_stories):
    super().__init__()
    self.rule_stories = rule_stories
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED, ignore_label=0)
      # self.embed = Embed()
      # self.rulegen = RuleGen()
      self.vmap_params = C.Parameter(0.0, (len(self.rule_stories), len(word2idx)), name='vmap_params')
      self.mematt = MemAttention()
      # self.mematt = MemN2N()
      self.uni_birnn = L.NStepBiGRU(1, EMBED, EMBED, DROPOUT)
      self.uni_linear = L.Linear(EMBED, EMBED, nobias=True)
      self.rule_linear = L.Linear(EMBED, EMBED, nobias=True)
      self.answer_linear = L.Linear(EMBED, len(word2idx))
    # Setup rule repo
    self.vrules = vectorise_stories(rule_stories) # (R, Ls, L), (R, Q), (R, A), (R, I)
    self.mrules = tuple([v != 0 for v in self.vrules[:-1]]) # (R, Ls, L), (R, Q), (R, A)
    self.log = None

  def tolog(self, key, value):
    """Append to log dictionary given key value pair."""
    loglist = self.log.setdefault(key, [])
    loglist.append(value)

  def compute_vmap(self):
    """Compute the variable map for rules."""
    rvctx, rvq, rva, rsupps = self.vrules # (R, Ls, L), (R, Q), (R, A), (R, I)
    # vmap = self.rulegen(self.vrules, [erctx, erq, era]) # (R, V)
    rwords = np.reshape(rvctx, (rvctx.shape[0], -1)) # (R, Ls*L)
    rwords = np.concatenate([rvq, rwords], -1) # (R, Q+Ls*L)
    wordrange = np.arange(len(word2idx)) # (V,)
    embedded_words = self.embed(wordrange) # (V, E)
    wordrange[0] = -1 # Null padding is never a variable
    mask = np.vstack([np.isin(wordrange, rws) for rws in rwords]) # (R, V)
    vmap = F.sigmoid(self.vmap_params*10) # (R, V)
    vmap *= mask # (R, V)
    self.tolog('vmap', vmap)
    return vmap

  def unify(self, toprove, embedded_toprove, candidates, embedded_candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (R, Ps, P)
    # embedded_toprove.shape = (R, Ps, P, E)
    # candidates.shape = (B, Cs, C)
    # embedded_candidates.shape = (B, Cs, C, E)
    # ---------------------------
    # Setup masks
    mask_toprove = (toprove != 0) # (R, Ps, P)
    mask_candidates = (candidates == 0) # (B, Cs, C)
    sim_mask = mask_candidates.astype(np.float32) * MINUS_INF # (B, Cs, C)
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    # contextual_toprove = contextual_convolve(self.xp, self.convolve_words, toprove, groundtoprove) # (R, Ps, P, E)
    # contextual_candidates = contextual_convolve(self.xp, self.convolve_words, candidates, embedded_candidates) # (B, Cs, C, E)
    # contextual_toprove = embedded_toprove # (R, Ps, P, E)
    contextual_toprove = seq_rnn_embed(toprove, embedded_toprove, self.uni_birnn, True) # (R, Ps, P, E)
    contextual_toprove = self.uni_linear(contextual_toprove, n_batch_axes=3) # (R, Ps, P, E)
    contextual_candidates = seq_rnn_embed(candidates, embedded_candidates, self.uni_birnn, True) # (B, Cs, C, E)
    contextual_candidates = self.uni_linear(contextual_candidates, n_batch_axes=3) # (B, Cs, C, E)
    # contextual_toprove = F.normalize(contextual_toprove, axis=-1) # (B, R, Ps, P, E)
    # contextual_candidates = F.normalize(contextual_candidates, axis=-1) # (B, Cs, C, E)
    # Compute similarity between every provable symbol and candidate symbol
    # (R, Ps, P, E) x (B, Cs, C, E)
    raw_sims = F.einsum("jklm,inom->ijklno", contextual_toprove, contextual_candidates) # (B, R, Ps, P, Cs, C)
    # raw_sims *= 10 # scale up for softmax
    # ---------------------------
    # Calculate attended unified word representations for toprove
    raw_sims += sim_mask[:, None, None, None, ...] # (B, R, Ps, P, Cs, C)
    sim_weights = F.softmax(raw_sims, -1) # (B, R, Ps, P, Cs, C)
    sim_weights *= mask_toprove[None, ..., None, None] # (B, R, Ps, P, Cs, C)
    # (B, R, Ps, P, Cs, C) x (B, Cs, C, E)
    unifications = F.einsum("ijklmn,imno->ijklmo", sim_weights, embedded_candidates) # (B, R, Ps, P, Cs, E)
    return unifications

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
    rmctx, rmq, rma = self.mrules # (R, Ls, L), (R, Q), (R, A)
    erctx, erq, era = [self.embed(v) for v in self.vrules[:-1]] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    # erctx, erq, era = erctx*rmctx[..., None], erq*rmq[..., None], era*rma[..., None] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    # ---------------------------
    # Compute variable map
    vmap = self.compute_vmap()
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    # vmap = np.array([[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)
    # vmap = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], dtype=np.float32)
    # self.tolog('vmap', vmap)
    # Setup variable maps and states
    nrules_range = np.arange(len(self.rule_stories)) # (R,)
    ctxvgates = vmap[nrules_range[:, None, None], rvctx] # (R, Ls, L)
    qvgates = vmap[nrules_range[:, None], rvq] # (R, Q)
    # ---------------------------
    # Rule states
    rs = self.mematt.init_state(rvq, erq) # (R, E)
    # Attention masks
    bodyattmask = np.all(rvctx == 0, -1) # (R, Ls)
    candattmask = np.all(vctx == 0, -1) # (B, Cs)
    # ---------------------------
    # Unify query first assuming given query is ground
    qunis = self.unify(rvq[:, None, :], erq[:, None, ...], vq[:, None, ...], eq[:, None, ...]) # (B, R, 1, Q, 1, E)
    qunis = F.squeeze(qunis, (2, 4)) # (B, R, Q, E)
    # ---------------------------
    # Get initial candidate state
    qstate = qvgates[..., None]*qunis + (1-qvgates[..., None])*erq # (B, R, Q, E)
    qstate *= rmq[..., None] # (B, R, Q, E)
    brvq = np.repeat(rvq[None, ...], qstate.shape[0], 0) # (B, R, Q)
    uni_cs = self.mematt.init_state(brvq, qstate) # (B, R, E)
    orig_cs = self.mematt.init_state(vq, eq) # (B, E)
    # ---------------------------
    # Compute rule attentions
    num_rules = rvq.shape[0] # R
    if num_rules > 1:
      cs_feats = self.rule_linear(orig_cs) # (B, E)
      ratt = cs_feats @ rs.T # (B, R)
      ratt = F.softmax(ratt, -1) # (B, R)
      self.tolog('ratt', ratt)
    # ---------------------------
    # Prepare unified state
    if num_rules == 1:
      uni_cs = uni_cs[:, 0] # (B, E)
    else:
      # (B, R) x (B, R, E)
      uni_cs = F.einsum('ij,ijk->ik', ratt, uni_cs) # (B, E)
    uniloss = F.mean_squared_error(uni_cs, orig_cs) # ()
    self.tolog('uniloss', uniloss)
    # ---------------------------
    # Unify body, every symbol to every symbol
    bunis = self.unify(rvctx, erctx, vctx, ectx) # (B, R, Ls, L, Cs, E)
    # ---------------------------
    # Setup memory sequence embeddings
    mem_erctx = self.mematt.seq_embed(rvctx, erctx) # (R, Ls, E)
    mem_ectx = self.mematt.seq_embed(vctx, ectx) # (B, Cs, E)
    # Compute iterative updates on variables
    for t in range(ITERATIONS):
      # ---------------------------
      # Compute which body literal to prove using rule state
      raw_body_att = self.mematt(rs, rvctx, mem_erctx, bodyattmask, t) # (R, Ls)
      self.tolog('raw_body_att', raw_body_att)
      body_att = F.softmax(raw_body_att, -1) # (R, Ls)
      # Compute unified candidate attention
      raw_uni_cands_att = self.mematt(uni_cs, vctx, mem_ectx, candattmask, t) # (B, Cs)
      self.tolog('raw_uni_cands_att', raw_uni_cands_att)
      uni_cands_att = F.softmax(raw_uni_cands_att, -1) # (B, Cs)
      # Compute original candidate attention
      raw_orig_cands_att = self.mematt(orig_cs, vctx, mem_ectx, candattmask, t) # (B, Cs)
      self.tolog('raw_orig_cands_att', raw_orig_cands_att)
      orig_cands_att = F.softmax(raw_orig_cands_att, -1) # (B, Cs)
      # ---------------------------
      # Update rule states
      rs = self.mematt.update_state(rs, body_att, rvctx, mem_erctx, t) # (R, E)
      # ---------------------------
      # Compute attended unification over candidates
      # (B, Cs) x (R, Ls, L) x (B, R, Ls, L, Cs, E)
      unis = F.einsum("im,jkl,ijklmo->ijklo", uni_cands_att, ctxvgates, bunis) # (B, R, Ls, L, E)
      # ---------------------------
      # Update candidate states with new variable bindings
      ground = (1-ctxvgates[..., None])*erctx # (R, Ls, L, E)
      bstate = ground + unis # (B, R, Ls, L, E)
      body_att = F.broadcast_to(body_att, bstate.shape[:3]) # (B, R, Ls)
      brvctx = np.repeat(rvctx[None, ...], bstate.shape[0], 0) # (B, R, Ls, L)
      mem_bstate = self.mematt.seq_embed(brvctx, bstate) # (B, R, Ls, E)
      uni_cs = F.repeat(uni_cs[:, None], num_rules, 1) # (B, R, E)
      uni_cs = self.mematt.update_state(uni_cs, body_att, brvctx, mem_bstate, t) # (B, R, E)
      orig_cs = self.mematt.update_state(orig_cs, orig_cands_att, vctx, mem_ectx, t) # (B, E)
      # Apply rule attention
      if num_rules == 1:
        uni_cs = uni_cs[:, 0] # (B, E)
      else:
        # (B, R) x (B, R, E)
        uni_cs = F.einsum('ij,ijk->ik', ratt, uni_cs) # (B, E)
      uniloss = F.mean_squared_error(uni_cs, orig_cs) # ()
      self.tolog('uniloss', uniloss)
    # ---------------------------
    # Compute answers based on variable and rule scores
    prediction = self.answer_linear(uni_cs) # (B, V)
    # Compute auxilary answers
    rpred = self.answer_linear(rs) # (R, V)
    rpredloss = F.softmax_cross_entropy(rpred, rva[:,0]) # ()
    self.tolog('rpredloss', rpredloss)
    opred = self.answer_linear(orig_cs) # (B, V)
    opredloss = F.softmax_cross_entropy(opred, va[:,0]) # ()
    self.tolog('opredloss', opredloss)
    oacc = F.accuracy(opred, va[:,0]) # ()
    self.tolog('oacc', oacc)
    # prediction = cs @ self.mematt.embedAC[-1].W.T # (B, V)
    return prediction

# ---------------------------

# Wrapper chain for training and predicting
class Classifier(C.Chain):
  """Compute loss and accuracy of underlying model."""
  def __init__(self, predictor):
    super().__init__()
    self.uniparam = 0.0
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
    uattloss = F.stack(self.predictor.log['raw_uni_cands_att'], 1) # (B, I, Cs)
    uattloss = F.hstack([F.softmax_cross_entropy(uattloss[:,i,:], supps[:,i]) for i in range(ITERATIONS)]) # (I,)
    uattloss = F.mean(uattloss) # ()
    # ---
    oattloss = F.stack(self.predictor.log['raw_orig_cands_att'], 1) # (B, I, Cs)
    oattloss = F.hstack([F.softmax_cross_entropy(oattloss[:,i,:], supps[:,i]) for i in range(ITERATIONS)]) # (I,)
    oattloss = F.mean(oattloss) # ()
    # ---
    rattloss = F.stack(self.predictor.log['raw_body_att'], 1) # (R, I, Ls)
    rattloss = F.hstack([F.softmax_cross_entropy(rattloss[:,i,:], rsupps[:,i]) for i in range(ITERATIONS)]) # (I,)
    rattloss = F.mean(rattloss) # ()
    # ---
    rpredloss = self.predictor.log['rpredloss'][0] # ()
    opredloss = self.predictor.log['opredloss'][0] # ()
    oacc = self.predictor.log['oacc'][0] # ()
    uniloss = F.hstack(self.predictor.log['uniloss']) # (I+1,)
    uniloss = F.mean(uniloss) # ()
    # ---
    C.report({'loss': mainloss, 'vmap': vmaploss, 'rpred': rpredloss, 'opred': opredloss, 'uni': uniloss, 'oacc': oacc, 'acc': acc}, self)
    if BABI:
      C.report({'uatt': uattloss, 'oatt': oattloss, 'ratt': rattloss}, self)
    return self.uniparam*(mainloss + 0.1*vmaploss + ARGS.strong*uattloss + uniloss) + ARGS.strong*(oattloss + rattloss) + opredloss + rpredloss # ()

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
optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))
# optimiser.add_hook(C.optimizer_hooks.GradientClipping(40))

train_iter = C.iterators.SerialIterator(train_enc_stories, 64)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas, supps = vectorise_stories(batch_stories, noise=False) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas, supps), vas[:, 0] # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (300, 'epoch'))

# Trainer extensions
def enable_unification(trainer):
  """Enable unification loss function in model."""
  trainer.updater.get_optimizer('main').target.uniparam = 1.0
trainer.extend(enable_unification, trigger=(20, 'epoch'))

def log_vmap(trainer):
  """Log inner properties to file."""
  vmaplog = trainer.updater.get_optimizer('main').target.predictor.log['vmap'][0] # (V,)
  logpath = os.path.join(trainer.out, ARGS.name + '_vmap.jsonl')
  with open(logpath, 'a') as f:
    f.write(str(trainer.updater.epoch) + "," + json.dumps(vmaplog.array.tolist()) + '\n')
trainer.extend(log_vmap, trigger=(1, 'epoch'))

# Validation extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1), name='val')
test_iter = C.iterators.SerialIterator(test_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(test_iter, cmodel, converter=converter, device=-1), name='test')
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_best.npz'), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(log_name=ARGS.name+'_log.json'))
# trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=ARGS.name+'_log.json'))
trainer.extend(T.extensions.FailOnNonNumber())
report_keys = ['loss', 'vmap', 'uatt', 'oatt', 'ratt', 'rpred', 'opred', 'uni', 'oacc', 'acc']
trainer.extend(T.extensions.PrintReport(['epoch'] + ['main/'+s for s in report_keys] + [p+'/main/'+s for p in ('val', 'test') for s in ('loss', 'acc')] + ['elapsed_time']))
# trainer.extend(T.extensions.ProgressBar(update_interval=10))
# trainer.extend(T.extensions.PlotReport(['main/loss', 'validation/main/loss'], 'iteration', marker=None, file_name=ARGS.name+'_loss.pdf'))
# trainer.extend(T.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'iteration', marker=None, file_name=ARGS.name+'_acc.pdf'))

# Setup training pausing
trainer_statef = trainer.out + '/' + ARGS.name + '_latest.npz'
def interrupt(signum, frame):
  """Save and interrupt training."""
  C.serializers.save_npz(trainer_statef, trainer)
  print("Getting interrupted, saved trainer file:", trainer_statef)
  raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)

# Check previously saved trainer
if os.path.isfile(trainer_statef):
  C.serializers.load_npz(trainer_statef, trainer)
  print("Loaded trainer state from:", trainer_statef)

# answer = model(converter([val_enc_stories[0]], None)[0])[0].array
# print("INIT ANSWER:", answer)
# Hit the train button
try:
  trainer.run()
except KeyboardInterrupt:
  pass

# Print final rules
answer = model(converter([test_enc_stories[0]], None)[0])[0].array
print("POST ANSWER:", answer)
print("RULE STORY:", [decode_story(rs) for rs in model.rule_stories])
print("ENC RULE STORY:", model.vrules)
print("RULE PARAMS:", model.log)
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
      print(model.log)
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
  for i in range(len(idx2word)):
    plt.annotate(idx2word[i], xy=(embds[i,0], embds[i,1]), xytext=(10, 10), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  plt.show()
