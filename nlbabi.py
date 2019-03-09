"""bAbI run on neurolog."""
import argparse
import logging
import os
import signal
import time
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
MAX_HIST = 25
REPO_SIZE = 1
ITERATIONS = 2
MINUS_INF = -100

# ---------------------------

def load_task(fname):
  """Load task stories from given file name."""
  ss = []
  with open(fname) as f:
    prev_id = 1
    context = list()
    for line in f:
      line = line.strip()
      sid, sl = line.split(' ', 1)
      # Is this a new story?
      sid = int(sid)-1
      if sid < prev_id:
        context = list()
      # Check for question or not
      if '\t' in sl:
        q, a, _ = sl.split('\t')
        cctx = context.copy()
        cctx.reverse()
        ss.append({'context': cctx[:MAX_HIST], 'query': q,
                        'answers': a.split(',')})
      else:
        # Just a statement
        context.append(sl)
      prev_id = sid
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
  for i, s in enumerate(encoded_stories):
    offset = 0
    vq[i,:len(s['query'])] = s['query']
    # who = s['query'][-1]
    # what = [c[-1] for c in s['context'] if c[0] == who]
    # target = [c[0] for c in s['context'] if c[-1] == what and c[0] != who]
    vas[i,:len(s['answers'])] = s['answers']
    # vas[i,:len(s['answers'])] = what
    for j, c in enumerate(s['context']):
      if offset < max_noise and np.random.rand() < 0.1:
        offset += 1
      vctx[i,j+offset,:len(c)] = c
  return vctx, vq, vas

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

def bow_encode(exs):
  """Given sentences compute is bag-of-words representation."""
  # [(s1len, E), (s2len, E), ...]
  return [F.sum(e, axis=0) for e in exs]

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
    vctx, vq, va = vectorised_rules
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
      self.words_linear = L.Linear(EMBED, EMBED)#, initialW=C.initializers.Orthogonal())

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
    with self.init_scope():
      self.att_linear = L.Linear(6*EMBED, 3*EMBED)
      self.att_score = L.Linear(3*EMBED, 1)
      self.temporal_enc = C.Parameter(C.initializers.Normal(1.0), (MAX_HIST, EMBED), name="tempenc")
      self.state_linear = L.Linear(EMBED, EMBED)

  def forward(self, equery, ememory, mask=None):
    """Compute an attention over memory given the query."""
    # equery.shape == (..., E)
    # ememory.shape == (..., Ms, E)
    # mask.shape == (..., Ms)
    # Setup mask
    nbatch = len(ememory.shape) - 2 # ...+1
    temps = self.temporal_enc[:ememory.shape[-2], :] # (Ms, E)
    temps = F.reshape(temps, [1]*nbatch+list(temps.shape)) # (1*..., Ms, E)
    evq, emem, temps = F.broadcast(equery[..., None, :], ememory, temps) # (..., Ms, E)
    att = F.concat([evq, emem, temps, evq*emem, temps*emem, temps+emem], -1) # (..., Ms, 6*E)
    att = self.att_linear(att, n_batch_axes=nbatch+1) # (..., Ms, 3*E)
    att = F.tanh(att) # (..., Ms, 3*E)
    att = self.att_score(att, n_batch_axes=nbatch+1) # (..., Ms, 1)
    att = F.squeeze(att, -1) # (..., Ms)
    if mask is not None:
      att += mask * MINUS_INF # (..., Ms)
    att = F.softmax(att, -1) # (..., Ms)
    return att

  def update_state(self, oldstate, state_att, newstates):
    """Update state given old, attention and new possible states."""
    # oldstate.shape == (..., E)
    # state_att.shape == (..., Ms)
    # newstates.shape == (..., Ms, E)
    # (..., Ms) x (..., Ms, E)
    ns = F.einsum("...i,...ij->...j", state_att, newstates) # (..., E)
    # concat = F.concat([oldstate, ns, oldstate*ns], -1) # (..., 3*E)
    # new_state = self.state_linear(concat, n_batch_axes=len(oldstate.shape)-1) # (..., E)
    hs = self.state_linear(oldstate, n_batch_axes=len(oldstate.shape)-1) # (..., E)
    new_state = hs + ns # (..., E)
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
      # self.embed = L.EmbedID(len(word2idx), EMBED, initialW=np.eye(len(word2idx)), ignore_label=0)
      # self.embed = Embed()
      self.rulegen = RuleGen()
      self.unify = Unify()
      self.mematt = MemAttention()
      self.embedA_linear = L.Linear(EMBED, EMBED)
      self.embedB_linear = L.Linear(EMBED, EMBED)
      self.embedC_linear = L.Linear(EMBED, EMBED)
      # self.rule_state_gate = L.Linear(EMBED, 1)
      self.rule_linear = L.Linear(8*EMBED, 4*EMBED)
      self.rule_score = L.Linear(4*EMBED, 1)
      self.answer_linear = L.Linear(EMBED, len(word2idx))
    # Setup rule repo
    self.eye = self.xp.eye(len(word2idx), dtype=self.xp.float32) # (V, V)
    self.vrules = vectorise_stories(rule_stories) # (R, Ls, L), (R, Q), (R, A)
    self.mrules = tuple([v != 0 for v in self.vrules]) # (R, Ls, L), (R, Q), (R, A)
    self.atts = None

  def gen_rules(self):
    """Generate the body and variable maps from rules in repository."""
    erctx, erq, era = [self.embed(v) for v in self.vrules] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    vmap = self.rulegen(self.vrules, [erctx, erq, era]) # (R, V)
    # inbody = np.zeros((9,2), dtype=np.float32)
    # inbody[0,0] = 1
    # inbody[2,0] = 1
    # inbody[3,0] = 1
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    # inbody = np.array([[0,0],[1,0]], dtype=np.float32)
    # vmap = np.array([[0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)
    return self.atts, vmap

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    self.atts = {k:list() for k in ('bodyatts', 'candsatt', 'rulegates')}
    # ---------------------------
    vctx, vq, va = stories # (B, Cs, C), (B, Q), (B, A)
    # Embed stories
    ectx = self.embed(vctx) # (B, Cs, C, E)
    ectxA = self.embedA_linear(ectx, n_batch_axes=3) # (B, Cs, C, E)
    ectxA = pos_encode(vctx, ectxA) # (B, Cs, E)
    eq = self.embed(vq) # (B, Q, E)
    # ---------------------------
    # Prepare rules and variable states
    rvctx, rvq, rva = self.vrules # (R, Ls, L), (R, Q), (R, A)
    rmctx, rmq, rma = self.mrules # (R, Ls, L), (R, Q), (R, A)
    erctx, erq, era = [self.embed(v) for v in self.vrules] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    erctxA = self.embedA_linear(erctx, n_batch_axes=3) # (R, Ls, L, E)
    erctxA = pos_encode(rvctx, erctxA) # (R, Ls, E)
    erctxC = self.embedC_linear(erctx, n_batch_axes=3) # (R, Ls, L, E)
    erctxC = pos_encode(rvctx, erctxC) # (R, Ls, E)
    # erctx, erq, era = erctx*rmctx[..., None], erq*rmq[..., None], era*rma[..., None] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    vmap = self.rulegen(self.vrules, [erctx, erq, era]) # (R, V)
    # Setup variable maps and states
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    # vmap = np.array([[0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)
    nrules_range = np.arange(len(self.rule_stories)) # (R,)
    ctxvgates = vmap[nrules_range[:, None, None], rvctx] # (R, Ls, L)
    qvgates, avgates = [vmap[nrules_range[:, None], v] for v in (rvq, rva)] # (R, Q), (R, A)
    # Rule states
    rs = self.embedB_linear(erq, n_batch_axes=2) # (R, Q, E)
    rs = pos_encode(rvq, rs) # (R, E)
    bodyattmask = np.all(rvctx == 0, -1) # (R, Ls)
    # Candidate states
    candattmask = np.all(vctx == 0, -1) # (B, Cs)
    # Helper ranges for indexing
    wordsrange = np.arange(len(word2idx)) # (V,)
    batchrange = np.arange(vctx.shape[0]) # (B,)
    bodyrange = np.arange(rvctx.shape[1]) # (Ls,)
    # ---------------------------
    # Unify query first assuming given query is ground
    qunis = self.unify(rvq[:, None, :], erq[:, None, ...], vq[:, None, ...], eq[:, None, ...]) # (B, R, 1, Q, 1, E)
    qunis = F.squeeze(qunis, (2, 4)) # (B, R, Q, E)
    # ---------------------------
    # Get initial candidate state
    qstate = qvgates[..., None]*qunis + (1-qvgates[..., None])*erq # (B, R, Q, E)
    qstate *= rmq[..., None] # (B, R, Q, E)
    cs = self.embedB_linear(qstate, n_batch_axes=3) # (B, R, Q, E)
    cs = pos_encode(vq[:, None, :], cs) # (B, R, E)
    # Compute iterative updates on variables
    # dummy_att = np.zeros((3, 1, 9), dtype=np.float32)
    # dummy_att[0,0,0] = 1
    # dummy_att[1,0,3] = 1
    # dummy_att[2,0,2] = 1
    for t in range(ITERATIONS):
      # ---------------------------
      # Unify body with updated variable state
      bunis = self.unify(rvctx, erctx, vctx, ectx) # (B, R, Ls, L, Cs, E)
      # ---------------------------
      # Compute which body literal to prove using rule state
      body_att = self.mematt(rs, erctxA, bodyattmask) # (R, Ls)
      # body_att = dummy_att[t] # (R, Ls)
      self.atts['bodyatts'].append(body_att)
      # Compute candidate attentions
      cands_att = self.mematt(cs, ectxA[:, None, ...], candattmask[:, None, ...]) # (B, R, Cs)
      self.atts['candsatt'].append(cands_att)
      # ---------------------------
      # Update rule states
      rs = self.mematt.update_state(rs, body_att, erctxC) # (R, E)
      # ---------------------------
      # Compute attended unification over candidates
      # (B, R, Cs) x (R, Ls, L) x (R, Ls, L) x (B, R, Ls, L, Cs, E)
      unis = F.einsum("ijm,jkl,jkl,ijklmo->ijklo", cands_att, rmctx.astype(np.float32), ctxvgates, bunis) # (B, R, Ls, L, E)
      # ---------------------------
      # Update candidate states with new variable bindings
      ground = (1-ctxvgates[..., None])*erctx # (R, Ls, L, E)
      bstate = ground + unis # (B, R, Ls, L, E)
      new_cs = self.embedC_linear(bstate, n_batch_axes=4) # (B, R, Ls, L, E)
      new_cs = pos_encode(rvctx, new_cs) # (B, R, Ls, E)
      att = F.repeat(body_att[None, ...], new_cs.shape[0], 0) # (B, R, Ls)
      cs = self.mematt.update_state(cs, att, new_cs) # (B, R, E)
    # ---------------------------
    # Compute answers based on variable and rule scores
    prediction = self.answer_linear(cs, n_batch_axes=2) # (B, R, V)
    # ---------------------------
    # Compute aux losses
    vmap_loss = F.sum(vmap) # ()
    aux_rloss = self.answer_linear(rs) # (R, V)
    aux_rloss = F.softmax_cross_entropy(aux_rloss, rva[:,0]) # ()
    # ---------------------------
    # Compute rule attentions
    num_rules = rvq.shape[0] # R
    if num_rules == 1:
      return prediction[:, 0, :], 0.01*(vmap_loss+aux_rloss) # (B, V), ()
    num_batch = vq.shape[0] # B
    # Rule features
    rqfeats = pos_encode(rvq, erq) # (R, E)
    rbfeats = pos_encode(rvctx, erctx) # (R, Ls, E)
    rbfeats = F.sum(rbfeats, -2) # (R, E)
    rfeats = F.concat([rqfeats, rbfeats], -1) # (R, 2*E)
    # Story features
    qfeats = pos_encode(vq, eq) # (B, E)
    bfeats = pos_encode(vctx, ectx) # (B, Cs, E)
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
    # ---------------------------
    # Compute final rule attended answer
    # (B, R) x (B, R, A, V)
    final_answers = F.einsum("ij,ijkl->ikl", rule_atts, answers) # (B, A, V)
    # **Assuming one answer for now***
    return final_answers[:, 0, :], F.sum(vmap)*0.01

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
def lossfun(outputs, targets):
  """Loss function for target classification."""
  # output.shape == (B, V)
  # targets.shape == (B,)
  predictions, auxloss = outputs # (B, V), ()
  classloss = F.softmax_cross_entropy(predictions, targets)
  return classloss + auxloss
def accfun(outputs, targets):
  """Compute classification accuracy."""
  return F.accuracy(outputs[0], targets)
cmodel = L.Classifier(model, lossfun=lossfun, accfun=accfun)

optimiser = C.optimizers.Adam().setup(cmodel)
optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))

train_iter = C.iterators.SerialIterator(enc_stories, 7)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas = vectorise_stories(batch_stories, noise=False) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas), vas[:, 0] # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (50, 'epoch'))

# Trainer extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 128, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_best.npz'), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(log_name=ARGS.name+'_log.json'))
# trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=ARGS.name+'_log.json'))
trainer.extend(T.extensions.FailOnNonNumber())
trainer.extend(T.extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
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
print("RULE PARAMS:", model.gen_rules())
# Extra inspection if we are debugging
if ARGS.debug:
  for val_story in val_enc_stories:
    answer = model(converter([val_story], None)[0])[0].array
    prediction = np.argmax(answer)
    expected = val_story['answers'][0]
    if prediction != expected:
      print(decode_story(val_story))
      print(f"Expected {expected} '{idx2word[expected]}' got {prediction} '{idx2word[prediction]}'.")
      break
  # Plot Embeddings
  # pca = PCA(2)
  # print(model.unify.words_linear.W.array.T)
  # embds = pca.fit_transform(model.unify.words_linear.W.array.T)
  # print("PCA VAR:", pca.explained_variance_ratio_)
  # plt.scatter(embds[:, 0], embds[:, 1])
  # for i in range(len(idx2word)):
    # plt.annotate(idx2word[i], xy=(embds[i,0], embds[i,1]), xytext=(10, 10), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  # plt.show()
  # Dig in
  import ipdb; ipdb.set_trace()
  answer = model(converter([val_story], None)[0])
