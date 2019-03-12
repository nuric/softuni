"""bAbI run on neurolog."""
import argparse
import logging
import os
import signal
import time
from collections import OrderedDict
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F
import chainer.training as T
import matplotlib
# matplotlib.use('pdf')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Disable scientific printing
np.set_printoptions(suppress=True, precision=3, linewidth=180)

# Arguments
parser = argparse.ArgumentParser(description="Run NeuroLog on bAbI tasks.")
parser.add_argument("task", help="File that contains task train.")
parser.add_argument("--name", default="nlbabi", help="Name prefix for saving files etc.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
ARGS = parser.parse_args()

# Debug
# if ARGS.debug:
  # logging.basicConfig(level=logging.DEBUG)
  # C.set_debug(True)

EMBED = 16
MAX_HIST = 200
REPO_SIZE = 1
ITERATIONS = 2
MINUS_INF = -100

# ---------------------------

def load_task(fname):
  """Load task stories from given file name."""
  ss = []
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
        assert len(supps) == ITERATIONS, "Not enough iterations for supporting facts."
        cctx = list(context.values())
        # cctx.reverse()
        ss.append({'context': cctx[:MAX_HIST], 'query': q,
                   'answers': a.split(','), 'supps': supps})
      else:
        # Just a statement
        context[sid] = sl
  return ss
stories = load_task(ARGS.task)
val_stories = load_task(ARGS.task.replace('train', 'test'))
print("TRAIN:", len(stories), "stories")
print("VAL:", len(val_stories), "stories")
print("SAMPLE:", stories[0])

# ---------------------------

# Tokenisation of sentences
def tokenise(text, filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n', split=' '):
  """Lower case naive space based tokeniser."""
  text = text.lower()
  translate_dict = dict((c, split) for c in filters)
  translate_map = str.maketrans(translate_dict)
  text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]

# Word indices
word2idx = {'unk':0}

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
val_enc_stories = list(map(encode_story, val_stories))
print("VAL VOCAB:", len(word2idx))
print("ENC SAMPLE:", enc_stories[0])
idx2word = {v:k for k, v in word2idx.items()}

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
  max_noise = 0
  if noise:
    max_noise = max_ctxlen // 10 # 10 percent noise
  # Vectorise stories
  vctx = np.zeros((len(encoded_stories), max_ctxlen+max_noise, ctx_maxlen), dtype=np.int32) # (B, Cs, C)
  vq = np.zeros((len(encoded_stories), q_maxlen), dtype=np.int32) # (B, Q)
  vas = np.zeros((len(encoded_stories), a_maxlen), dtype=np.int32) # (B, A)
  supps = np.zeros((len(encoded_stories), ITERATIONS), dtype=np.int32) # (B, I)
  for i, s in enumerate(encoded_stories):
    offset = 0
    vq[i,:len(s['query'])] = s['query']
    # who = s['query'][-1]
    # what = [c[-1] for c in s['context'] if c[0] == who]
    # target = [c[0] for c in s['context'] if c[-1] == what and c[0] != who]
    vas[i,:len(s['answers'])] = s['answers']
    # vas[i,:len(s['answers'])] = what
    supps[i] = s['supps']
    for j, c in enumerate(s['context']):
      if offset < max_noise and np.random.rand() < 0.1:
        offset += 1
      vctx[i,j+offset,:len(c)] = c
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

def story_embed(story, embed):
  """Given an encoded story embed its sentences using embed."""
  encs = sequence_embed([story['answers'], story['query']] + story['context'], embed)
  return {'answers': encs[0], 'query': encs[1], 'context': encs[2:]}

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

def seq_encode(vxs, exs):
  """Encode a sequence."""
  # (..., S), (..., S, E)
  return bow_encode(vxs, exs)

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

# ---------------------------

# Rule generating network
class RuleGen(C.Chain):
  """Takes an example story-> context, query, answer
  returns a probabilistic rule"""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.body_linear = L.Linear(8*EMBED, 4*EMBED)
      self.body_score = L.Linear(4*EMBED, 2)
      # self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.isvariable_linear = L.Linear(EMBED+1, EMBED)
      self.isvariable_score = L.Linear(EMBED, 1)

  def forward(self, vectorised_rules, embedded_rules):
    """Given a story generate a probabilistic learnable rule."""
    # vectorised_rules = [(R, Ls, L), (R, Q), (R, A)]
    # embedded_rules = [(R, Ls, L, E), (R, Q, E), (R, A, E)]
    vctx, vq, va, _ = vectorised_rules
    ectx, eq, ea = embedded_rules
    # ---------------------------
    # Whether each word in story is a variable, binary class -> {wordid:sigmoid}
    num_rules = vctx.shape[0] # R
    words = np.reshape(vctx, (num_rules, -1)) # (R, Ls*L)
    words = np.concatenate([vq, words], -1) # (R, Q+Ls*L)
    # Compute contextuals by convolving each sentence
    ###
    # contextual_q = contextual_convolve(self.xp, self.convolve_words, vq, eq) # (R, Q, E)
    # contextual_ctx = contextual_convolve(self.xp, self.convolve_words, vctx, ectx) # (R, Ls, L, E)
    # flat_cctx = F.reshape(contextual_ctx, (num_rules, -1, ectx.shape[-1])) # (R, Ls * L, E)
    # cwords = F.concat([contextual_q, flat_cctx], 1) # (R, Q+Ls*L, E)
    ###
    flat_ctx = F.reshape(ectx, (num_rules, -1, ectx.shape[-1])) # (R, Ls * L, E)
    cwords = F.concat([eq, flat_ctx], 1) # (R, Q+Ls*L, E)
    # Add whether they appear in the answer as a featurec
    # np.isin flattens second argument, so we need for loop
    appearanswer = np.array([np.isin(ws, _va) for ws, _va in zip(words, va)]) # (R, Q+Ls*L)
    appearanswer = appearanswer.astype(np.float32) # (R, Q+Ls*L)
    allwords = F.concat([cwords, appearanswer[..., None]], -1) # (R, Q+Ls*L, E+1)
    wordvars = self.isvariable_linear(allwords, n_batch_axes=2) # (R, Q+Ls*L, E)
    wordvars = F.tanh(wordvars) # (R, Q+Ls*L, E)
    wordvars = self.isvariable_score(wordvars, n_batch_axes=2) # (R, Q+Ls*L, 1)
    wordvars = F.squeeze(wordvars, -1) # (R, Q+Ls*L)
    # Merge word variable predictions
    iswordvar = self.xp.zeros((num_rules, len(word2idx)), dtype=self.xp.float32) # (R, V)
    iswordvar = F.scatter_add(iswordvar, (np.arange(num_rules)[:, None], words), wordvars) # (R, V)
    iswordvar = F.sigmoid(iswordvar) # (R, V)
    wordrange = np.arange(len(word2idx)) # (V,)
    wordrange[0] = -1 # Null padding is never a variable
    # np.isin flattens second argument, so we need for loop
    mask = np.vstack([np.isin(wordrange, rws) for rws in words]) # (R, V)
    iswordvar *= mask # (R, V)
    # ---------------------------
    # Tells whether a word is a variable or not
    return iswordvar

# ---------------------------

# Unification network
class Unify(C.Chain):
  """Semantic unification on two sentences with variables."""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      # self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.words_linear = L.Linear(EMBED, EMBED, initialW=C.initializers.Orthogonal())

  def forward(self, toprove, embedded_toprove, candidates, embedded_candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (R, Ps, P)
    # embedded_toprove.shape = (R, Ps, P, E)
    # candidates.shape = (B, Cs, C)
    # embedded_candidates.shape = (B, Cs, C, E)
    # vstate.shape = (B, R, V, V)
    # ---------------------------
    # Setup masks
    mask_toprove = (toprove == 0.0) # (R, Ps, P)
    mask_candidates = (candidates == 0.0) # (B, Cs, C)
    sim_mask = np.logical_or(mask_toprove[None, ..., None, None], mask_candidates[:, None, None, None, ...]) # (B, R, Ps, P, Cs, C)
    sim_mask = sim_mask.astype(np.float32) * MINUS_INF # (B, R, Ps, P, Cs, C)
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    # contextual_toprove = contextual_convolve(self.xp, self.convolve_words, toprove, groundtoprove) # (R, Ps, P, E)
    # contextual_candidates = contextual_convolve(self.xp, self.convolve_words, candidates, embedded_candidates) # (B, Cs, C, E)
    contextual_toprove = self.words_linear(embedded_toprove, n_batch_axes=3) # (R, Ps, P, E)
    contextual_candidates = self.words_linear(embedded_candidates, n_batch_axes=3) # (B, Cs, C, E)
    # contextual_toprove = F.normalize(contextual_toprove, axis=-1) # (B, R, Ps, P, E)
    # contextual_candidates = F.normalize(contextual_candidates, axis=-1) # (B, Cs, C, E)
    # Compute similarity between every provable symbol and candidate symbol
    # (R, Ps, P, E) x (B, Cs, C, E)
    raw_sims = F.einsum("jklm,inom->ijklno", contextual_toprove, contextual_candidates) # (B, R, Ps, P, Cs, C)
    raw_sims += sim_mask # (B, R, Ps, P, Cs, C)
    # raw_sims *= 10 # scale up for softmax
    # ---------------------------
    # Calculate attended unified word representations for toprove
    sim_weights = F.softmax(raw_sims, -1) # (B, R, Ps, P, Cs, C)
    # (B, R, Ps, P, Cs, C) x (B, Cs, C, E)
    unifications = F.einsum("ijklmn,imno->ijklmo", sim_weights, embedded_candidates) # (B, R, Ps, P, Cs, E)
    return unifications

# ---------------------------

# Memory querying component
class MemAttention(C.Chain):
  """Computes attention over memory components given query."""
  def __init__(self):
    super().__init__()
    self.drop = 0.1
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED)
      # self.word_mask = L.Linear(EMBED, 1)
      self.q_linear = L.Linear(EMBED, EMBED)
      self.mem_linear = L.Linear(EMBED, EMBED)
      self.temporal = L.EmbedID(MAX_HIST, EMBED)
      self.att_linear = L.Linear(4*EMBED, EMBED)
      self.att_score = L.Linear(EMBED, 1)
      self.state_linear = L.Linear(3*EMBED, EMBED)

  def seq_embed(self, vxs):
    """Embed a given sequence."""
    # vxs.shape == (..., S)
    exs = self.embed(vxs) # (..., S, E)
    # mask = self.word_mask(exs, n_batch_axes=len(vxs.shape)) # (..., S, 1)
    # mask = F.sigmoid(mask) # (..., S, 1)
    mask = (vxs != 0)[..., None] # (..., S, 1)
    exs *= mask # (..., S, E)
    return F.sum(exs, -2) # (..., E)

  def init_state(self, vxs):
    """Initialise given state."""
    # vxs.shape == (..., S)
    s = self.seq_embed(vxs) # (..., E)
    s = self.q_linear(s, n_batch_axes=len(vxs.shape)-1) # (..., E)
    s = F.tanh(s) # (..., E)
    s = F.dropout(s, self.drop) # (..., E)
    return s # (..., E)

  def self_att(self, ememory, mask=None):
    """Compute self attention over embedded memory."""
    # ememory.shape == (..., Ms, E)
    # mask.shape == (..., Ms)
    keys = self.mem_linear(ememory, n_batch_axes=len(ememory.shape)-1) # (..., Ms, E)
    att = F.einsum("...ik,...jk->...ij", keys, ememory) # (..., Ms, Ms)
    if mask is not None:
      att += mask[..., None, :] * MINUS_INF # (..., Ms, Ms)
    att = F.softmax(att, -1) # (..., Ms, Ms)
    attended = F.einsum("...ij,...jk->...ik", att, ememory) # (..., Ms, E)
    return attended

  def forward(self, equery, vmemory, mask=None, iteration=0):
    """Compute an attention over memory given the query."""
    # equery.shape == (..., E)
    # vmemory.shape == (..., Ms, M)
    # mask.shape == (..., Ms)
    # Setup memory embedding
    ememory = self.seq_embed(vmemory) # (..., Ms, E)
    eq, em = F.broadcast(equery[..., None, :], ememory) # (..., Ms, E)
    # sem = self.self_att(ememory, mask) # (..., Ms, E)
    merged = F.concat([eq, em, eq*em, F.squared_difference(eq, em)], -1) # (..., Ms, 4*E)
    inter = self.att_linear(merged, n_batch_axes=len(vmemory.shape)-1) # (..., Ms, E)
    inter = F.tanh(inter) # (..., Ms, E)
    tidxs = self.xp.arange(vmemory.shape[-2], dtype=self.xp.int32) # (Ms,)
    temps = self.temporal(tidxs) # (Ms, E)
    inter += temps # (..., Ms, E)
    inter = F.dropout(inter, self.drop) # (..., Ms, E)
    att = self.att_score(inter, n_batch_axes=len(vmemory.shape)-1) # (..., Ms, 1)
    att = F.squeeze(att, -1) # (..., Ms)
    if mask is not None:
      att += mask * MINUS_INF # (..., Ms)
    # att = F.softmax(att, -1) # (..., Ms)
    return att

  def update_state(self, oldstate, mem_att, vmemory, iteration=0):
    """Update state given old, attention and new possible states."""
    # oldstate.shape == (..., E)
    # mem_att.shape == (..., Ms)
    # vmemory.shape == (..., Ms, M)
    # Setup memory output embedding
    ememory = self.seq_embed(vmemory) # (..., Ms, E)
    os, em = F.broadcast(oldstate[..., None, :], ememory) # (..., Ms, E)
    merged = F.concat([os, em, os+em], -1) # (..., Ms, 3*E)
    # TODO(nuric): add temporal on state updating
    tidxs = self.xp.arange(vmemory.shape[-2], dtype=self.xp.int32) # (Ms,)
    temps = self.temporal(tidxs) # (Ms, E)
    new_states = self.state_linear(merged, n_batch_axes=len(vmemory.shape)-1) # (..., Ms, E)
    new_states = F.tanh(new_states) # (..., Ms, E)
    # (..., Ms) x (..., Ms, E) -> (..., E)
    new_state = F.einsum("...i,...ij->...j", mem_att, new_states) # (..., E)
    new_state = F.dropout(new_state, self.drop) # (..., E)
    return new_state

# ---------------------------
class Embed:
  W = np.eye(len(word2idx), dtype=np.float32)
  def __call__(self, x):
    return F.embed_id(x, self.W, ignore_label=0)

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
      self.rulegen = RuleGen()
      self.unify = Unify()
      self.mematt = MemAttention()
      # self.rule_state_gate = L.Linear(EMBED, 1)
      self.rule_linear = L.Linear(8*EMBED, 4*EMBED)
      self.rule_score = L.Linear(4*EMBED, 1)
      self.answer_linear = L.Linear(EMBED, len(word2idx))
    # Setup rule repo
    self.eye = self.xp.eye(len(word2idx), dtype=self.xp.float32) # (V, V)
    self.vrules = vectorise_stories(rule_stories) # (R, Ls, L), (R, Q), (R, A)
    self.mrules = tuple([v != 0 for v in self.vrules]) # (R, Ls, L), (R, Q), (R, A)
    self.log = None

  def get_log(self):
    """Return last mini-batch log."""
    return self.log

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    self.log = {k:list() for k in ('vmap', 'bodyatts', 'candsatt', 'rule_atts')}
    # ---------------------------
    vctx, vq, va, supps = stories # (B, Cs, C), (B, Q), (B, A), (B, I)
    # Embed stories
    # ectx = self.embed(vctx) # (B, Cs, C, E)
    # eq = self.embed(vq) # (B, Q, E)
    # ---------------------------
    # Prepare rules and variable states
    rvctx, rvq, rva, rsupps = self.vrules # (R, Ls, L), (R, Q), (R, A)
    # rmctx, rmq, rma = self.mrules # (R, Ls, L), (R, Q), (R, A)
    erctx, erq, era = [self.embed(v) for v in self.vrules[:-1]] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    # erctx, erq, era = erctx*rmctx[..., None], erq*rmq[..., None], era*rma[..., None] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    vmap = self.rulegen(self.vrules, [erctx, erq, era]) # (R, V)
    self.log['vmap'] = vmap
    # Setup variable maps and states
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    # vmap = np.array([[0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)
    # nrules_range = np.arange(len(self.rule_stories)) # (R,)
    # ctxvgates = vmap[nrules_range[:, None, None], rvctx] # (R, Ls, L)
    # qvgates, avgates = [vmap[nrules_range[:, None], v] for v in (rvq, rva)] # (R, Q), (R, A)
    # Rule states
    # rs = self.mematt.init_state(rvq) # (R, E)
    bodyattmask = np.all(rvctx == 0, -1) # (R, Ls)
    # Candidate states
    candattmask = np.all(vctx == 0, -1) # (B, Cs)
    # ---------------------------
    # Unify query first assuming given query is ground
    # qunis = self.unify(rvq[:, None, :], erq[:, None, ...], vq[:, None, ...], eq[:, None, ...]) # (B, R, 1, Q, 1, E)
    # qunis = F.squeeze(qunis, (2, 4)) # (B, R, Q, E)
    # ---------------------------
    # Get initial candidate state
    # qstate = qvgates[..., None]*qunis + (1-qvgates[..., None])*erq # (B, R, Q, E)
    # qstate *= rmq[..., None] # (B, R, Q, E)
    # cs = seq_encode(vq[:, None, :], qstate) # (B, R, E)
    cs = self.mematt.init_state(vq) # (B, E)
    # Compute iterative updates on variables
    # dummy_att = np.zeros((3, 1, 9), dtype=np.float32)
    # dummy_att[0,0,0] = 1
    # dummy_att[1,0,3] = 1
    # dummy_att[2,0,2] = 1
    for t in range(ITERATIONS):
      # ---------------------------
      # Unify body with updated variable state
      # bunis = self.unify(rvctx, erctx, vctx, ectx) # (B, R, Ls, L, Cs, E)
      # ---------------------------
      # Compute which body literal to prove using rule state
      # body_att = self.mematt(rs, rvctx, bodyattmask, t) # (R, Ls)
      # body_att = dummy_att[t] # (R, Ls)
      # self.atts['bodyatts'].append(body_att)
      # Compute candidate attentions
      # cands_att = self.mematt(cs, pos_ectx[:, None, ...], candattmask[:, None, ...]) # (B, R, Cs)
      raw_cands_att = self.mematt(cs, vctx, candattmask, t) # (B, Cs)
      self.log['candsatt'].append(raw_cands_att)
      cands_att = F.softmax(raw_cands_att, -1) # (B, Cs)
      # ---------------------------
      # Update rule states
      # rs = self.mematt.update_state(rs, body_att, rvctx, t) # (R, E)
      # ---------------------------
      # Compute attended unification over candidates
      # (B, R, Cs) x (R, Ls, L) x (R, Ls, L) x (B, R, Ls, L, Cs, E)
      # unis = F.einsum("ijm,jkl,jkl,ijklmo->ijklo", cands_att, rmctx.astype(np.float32), ctxvgates, bunis) # (B, R, Ls, L, E)
      # ---------------------------
      # Update candidate states with new variable bindings
      # ground = (1-ctxvgates[..., None])*erctx # (R, Ls, L, E)
      # bstate = ground + unis # (B, R, Ls, L, E)
      # new_cs = seq_encode(rvctx, bstate) # (B, R, Ls, E)
      # att = F.repeat(body_att[None, ...], new_cs.shape[0], 0) # (B, R, Ls)
      # cs = self.mematt.update_state(cs, att, new_cs) # (B, R, E)
      cs = self.mematt.update_state(cs, cands_att, vctx, t) # (B, R, E)
    # ---------------------------
    # Compute answers based on variable and rule scores
    prediction = self.answer_linear(cs, n_batch_axes=1) # (B, R, V)
    # ---------------------------
    # Compute aux losses
    vmap_loss = F.sum(vmap) # ()
    # aux_rloss = self.answer_linear(rs) # (R, V)
    # aux_rloss = F.softmax_cross_entropy(aux_rloss, rva[:,0]) # ()
    attloss = F.stack(self.log['candsatt'], 1) # (B, I, Cs)
    attloss = F.hstack([F.softmax_cross_entropy(attloss[:,i,:], supps[:,i]) for i in range(ITERATIONS)]) # (I,)
    attloss = F.mean(attloss) # ()
    # ---------------------------
    # Compute rule attentions
    num_rules = rvq.shape[0] # R
    if num_rules == 1:
      # return prediction[:, 0, :], 0.01*(vmap_loss+aux_rloss) # (B, V), ()
      return prediction, 0.01*vmap_loss+attloss # (B, V), ()
    # Rule features
    rqfeats = seq_encode(rvq, erq) # (R, E)
    rbfeats = seq_encode(rvctx, erctx) # (R, Ls, E)
    rbfeats = F.sum(rbfeats, -2) # (R, E)
    rfeats = F.concat([rqfeats, rbfeats], -1) # (R, 2*E)
    # Story features
    qfeats = seq_encode(vq, eq) # (B, E)
    bfeats = seq_encode(vctx, ectx) # (B, Cs, E)
    bfeats = F.sum(bfeats, -2) # (B, E)
    feats = F.concat([qfeats, bfeats], -1) # (B, 2*E)
    # Compute attention score
    rfeats, feats = F.broadcast(rfeats[None, ...], feats[:, None, :]) # (B, R, 2*E)
    allfeats = F.concat([rfeats, feats, rfeats*feats, rfeats+feats], -1) # (B, R, 4*2*E)
    rule_atts = self.rule_linear(allfeats, n_batch_axes=2) # (B, R, 4*E)
    rule_atts = F.tanh(rule_atts) # (B, R, 4*E)
    rule_atts = self.rule_score(rule_atts, n_batch_axes=2) # (B, R, 1)
    rule_atts = F.squeeze(rule_atts, -1) # (B, R)
    rule_atts = F.softmax(rule_atts, -1) # (B, R)
    self.log['rule_atts'].append(rule_atts)
    # ---------------------------
    # Compute final rule attended answer
    # (B, R) x (B, R, V)
    final_prediction = F.einsum("ij,ijk->ik", rule_atts, prediction) # (B, V)
    return final_prediction, 0.01*(vmap_loss+aux_rloss) # (B, V), ()

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
    predictions, auxloss = self.predictor(xin) # (B, V), ()
    mainloss = F.softmax_cross_entropy(predictions, targets) # ()
    acc = F.accuracy(predictions, targets) # ()
    C.reporter.report({'loss': mainloss, 'aux': auxloss, 'acc': acc}, self)
    return mainloss + auxloss # ()

# ---------------------------

# Stories to generate rules from
answers = set()
rule_repo = list()
# np.random.shuffle(enc_stories)
for es in enc_stories:
  if es['answers'][0] not in answers:
    rule_repo.append(es)
    answers.add(es['answers'][0])
  if len(rule_repo) == REPO_SIZE:
    break
print("RULE REPO:", rule_repo)

# ---------------------------

# Setup model
model = Infer(rule_repo)
cmodel = Classifier(model)

optimiser = C.optimizers.Adam().setup(cmodel)
optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))
# optimiser.add_hook(C.optimizer_hooks.GradientClipping(40))

train_iter = C.iterators.SerialIterator(enc_stories, 64)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas, supps = vectorise_stories(batch_stories, noise=False) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas, supps), vas[:, 0] # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (200, 'epoch'))

# Trainer extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_best.npz'), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(log_name=ARGS.name+'_log.json'))
# trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=ARGS.name+'_log.json'))
trainer.extend(T.extensions.FailOnNonNumber())
trainer.extend(T.extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/aux', 'validation/main/aux', 'main/acc', 'validation/main/acc', 'elapsed_time']))
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
answer = model(converter([val_enc_stories[0]], None)[0])[0].array
print("POST ANSWER:", answer)
print("RULE STORY:", [decode_story(rs) for rs in model.rule_stories])
print("ENC RULE STORY:", model.vrules)
print("RULE PARAMS:", model.get_log())
# Extra inspection if we are debugging
if ARGS.debug:
  for val_story in val_enc_stories:
    answer, auxloss = model(converter([val_story], None)[0])
    prediction = np.argmax(answer.array)
    expected = val_story['answers'][0]
    if prediction != expected:
      print(decode_story(val_story))
      print(f"Expected {expected} '{idx2word[expected]}' got {prediction} '{idx2word[prediction]}'.")
      print(f"Aux loss: {auxloss}")
      print(model.get_log())
      import ipdb; ipdb.set_trace()
      answer = model(converter([val_story], None)[0])
  # Plot Embeddings
  # pca = PCA(2)
  # print(model.unify.words_linear.W.array.T)
  # embds = pca.fit_transform(model.unify.words_linear.W.array.T)
  # print("PCA VAR:", pca.explained_variance_ratio_)
  # plt.scatter(embds[:, 0], embds[:, 1])
  # for i in range(len(idx2word)):
    # plt.annotate(idx2word[i], xy=(embds[i,0], embds[i,1]), xytext=(10, 10), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  # plt.show()
