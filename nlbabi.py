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
# import matplotlib
# matplotlib.use('pdf')


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
ITERATIONS = 1
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

def vectorise_stories(encoded_stories):
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
  for i, s in enumerate(encoded_stories):
    vq[i,:len(s['query'])] = s['query']
    vas[i,:len(s['answers'])] = s['answers']
    for j, c in enumerate(s['context']):
      vctx[i,j,:len(c)] = c
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

# pe = np.zeros((20, EMBED), dtype=np.float32) # (20, E) 20 is max length
# for pos in range(pe.shape[0]):
  # for i in range(0, pe.shape[1], 2):
    # pe[pos, i] = np.sin(pos / (10000 ** ((2*i)/EMBED)))
    # pe[pos, i+1] = np.cos(pos / (10000 ** ((2*(i+1))/EMBED)))

def pos_encode(vxs, exs):
  """Given sentences compute positional encoding."""
  # (..., S), (..., S, E)
  mask = (vxs != 0.0) # (..., S)
  slen = exs.shape[-2] # S
  pos = np.fromfunction(lambda j, k: 1 - (j + 1) / slen - (k + 1) / EMBED * (1 - 2 * (j + 1) / slen),
                        (slen, EMBED), dtype=np.float32) # (S, E)
  enc = exs * pos # (..., S, E)
  enc *= mask[..., None] # (..., S, E)
  return F.sum(enc, -2) # (..., E)

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

  def forward(self, vectorised_rules, embedded_rules, temporal):
    """Given a story generate a probabilistic learnable rule."""
    # vectorised_rules = [(R, Ls, L), (R, Q), (R, A)]
    # embedded_rules = [(R, Ls, L, E), (R, Q, E), (R, A, E)]
    vctx, vq, va = vectorised_rules
    ectx, eq, ea = embedded_rules
    # Encode sequences
    enc_ctx = pos_encode(vctx, ectx) # (R, Ls, E)
    enc_query = pos_encode(vq, eq) # (R, E)
    enc_answer = pos_encode(va, ea) # (R, E)
    # ---------------------------
    # Whether a fact is in the body or not, negated or not, multi-label -> (clen, 2)
    bodylen = enc_ctx.shape[1] # Ls
    r_answer = F.repeat(enc_answer[:, None, :], bodylen, 1) # (R, Ls, E)
    r_query = F.repeat(enc_query[:, None, :], bodylen, 1) # (R, Ls, E)
    temps = temporal[:bodylen, :] # (Ls, E)
    temps = F.repeat(temps[None, ...], enc_ctx.shape[0], 0) # (R, Ls, E)
    r_ctx = F.concat([r_answer, r_query, enc_ctx, temps, temps+enc_ctx, temps*enc_ctx, r_answer*enc_ctx, r_query*enc_ctx], -1) # (R, Ls, 8*E)
    inbody = self.body_linear(r_ctx, n_batch_axes=2) # (R, Ls, 4*E)
    inbody = F.tanh(inbody) # (R, Ls, 4*E)
    inbody = self.body_score(inbody, n_batch_axes=2) # (R, Ls, 2)
    inbody = F.sigmoid(inbody) # (R, Ls, 2)
    bodymask = (vctx != 0.0) # (R, Ls, S)
    bodymask = np.any(bodymask, -1) # (R, Ls)
    inbody *= bodymask[..., None] # (R, Ls, 2)
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
    # Tells whether a context is in the body, negated or not
    # and whether a word is a variable or not
    return inbody, iswordvar

# ---------------------------

# Unification network
class Unify(C.Chain):
  """Semantic unification on two sentences with variables."""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      # self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.match_linear = L.Linear(6*EMBED, 3*EMBED)
      self.match_score = L.Linear(3*EMBED, 1)
    self.eye = self.xp.eye(len(word2idx), dtype=self.xp.float32) # (V, V)

  def forward(self, toprove, groundtoprove, vartoprove, candidates, embedded_candidates, temporal):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (R, Ps, P)
    # groundtoprove.shape = (R, Ps, P, E)
    # vartoprove.shape = (B, R, Ps, P, E)
    # candidates.shape = (B, Cs, C)
    # embedded_candidates.shape = (B, Cs, C, E)
    # vstate.shape = (B, R, V, V)
    # ---------------------------
    # Setup masks
    # mask_toprove = (toprove == 0.0) # (R, Ps, P)
    mask_candidates = (candidates == 0.0) # (B, Cs, C)
    # sim_mask = np.logical_or(mask_toprove[None, ..., None, None], mask_candidates[:, None, None, None, ...]) # (B, R, Ps, P, Cs, C)
    # sim_mask = sim_mask.astype(np.float32) * MINUS_INF # (B, R, Ps, P, Cs, C)
    mask_cs = np.all(mask_candidates, -1) # (B, Cs)
    mask_cs = mask_cs.astype(np.float32) * MINUS_INF # (B, Cs)
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    # contextual_toprove = contextual_convolve(self.xp, self.convolve_words, toprove, embedded_toprove) # (B, R, Ps, P, E)
    # contextual_candidates = contextual_convolve(self.xp, self.convolve_words, candidates, embedded_candidates) # (B, Cs, C, E)
    contextual_toprove = groundtoprove # (R, Ps, P, E)
    contextual_candidates = embedded_candidates # (B, Cs, C, E)
    # Compute similarity between every provable symbol and candidate symbol
    # (R, Ps, P, E) x (B, Cs, C, E)
    raw_sims = F.einsum("jklm,inom->ijklno", contextual_toprove, contextual_candidates) # (B, R, Ps, P, Cs, C)
    # raw_sims += sim_mask # (B, R, Ps, P, Cs, C)
    # ---------------------------
    # Calculate score for each candidate
    # Compute bag of words
    pbow = F.sum(vartoprove, -2) # (B, R, Ps, E)
    # pbow = pos_encode(toprove, contextual_toprove) # (B, R ,Ps, E)
    cbows = F.sum(embedded_candidates, -2) # (B, Cs, E)
    # cbows = pos_encode(candidates, contextual_candidates) # (B, Cs, E)
    tbows = temporal[:cbows.shape[-2], :] # (Cs, E)
    pbow, cbows, tbows = F.broadcast(pbow[..., None, :], cbows[:, None, None, ...], tbows[None, None, None, ...]) # (B, R, Ps, Cs, E)
    # Compute features
    pcbows = F.concat([pbow, cbows, pbow*cbows, tbows, tbows+cbows, tbows*cbows], -1) # (B, R, Ps, Cs, 6*E)
    raw_scores = self.match_linear(pcbows, n_batch_axes=4) # (B, R, Ps, Cs, 3*E)
    raw_scores = F.tanh(raw_scores) # (B, R, Ps, Cs, 3*E)
    raw_scores = self.match_score(raw_scores, n_batch_axes=4) # (B, R, Ps, Cs, 1)
    raw_scores = F.squeeze(raw_scores, -1) # (B, R, Ps, Cs)
    raw_scores += mask_cs[:, None, None, :] # (B, R, Ps, Cs)
    # ---------------------------
    # Calculate attended unified word representations for toprove
    unifications = self.eye[candidates] # (B, Cs, C, V)
    # (B, R, Ps, P, Cs, C) x (B, Cs, C, V)
    unifications = F.einsum("ijklmn,imno->ijklmo", raw_sims, unifications) # (B, R, Ps, P, Cs, V)
    # Weighted sum using scores
    weights = F.softmax(raw_scores, -1) # (B, R, Ps, Cs)
    # (B, R, Ps, Cs) x (B, R, Ps, P, Cs, V)
    final_unis = F.einsum("ijkc,ijklcv->ijklv", weights, unifications) # (B, R, Ps, P, V)
    # Final overall score
    final_scores = F.max(raw_scores, -1) # (B, R, Ps)
    final_scores *= np.any(toprove != 0.0, -1) # (B, R, Ps)
    return final_scores, final_unis

# ---------------------------

# Inference network
class Infer(C.Chain):
  """Takes a story, a set of rules and predicts answers."""
  def __init__(self, rule_stories):
    super().__init__()
    self.rule_stories = rule_stories
    # Create model parameters
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED, ignore_label=0)
      self.temporal_enc = C.Parameter(C.initializers.Normal(1.0), (MAX_HIST, EMBED), name="tempenc")
      self.rulegen = RuleGen()
      self.unify = Unify()
      self.unkbias = C.Parameter(6.0, shape=(1,), name="unkbias")
      self.eye = self.xp.eye(len(word2idx), dtype=self.xp.float32) # (V, V)
      self.rule_linear = L.Linear(8*EMBED, 4*EMBED)
      self.rule_score = L.Linear(4*EMBED, 1)
    # Setup rule repo
    self.vrules = vectorise_stories(rule_stories) # (R, Ls, L), (R, Q), (R, A)
    self.mrules = tuple([v != 0 for v in self.vrules]) # (R, Ls, L), (R, Q), (R, A)

  def gen_rules(self):
    """Generate the body and variable maps from rules in repository."""
    erctx, erq, era = [self.embed(v) for v in self.vrules] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    inbody, vmap = self.rulegen(self.vrules, [erctx, erq, era], self.temporal_enc) # (R, Ls, 2), (R, V)
    # inbody = np.zeros((9,2), dtype=np.float32)
    # inbody[0,0] = 1
    # inbody[2,0] = 1
    # inbody[3,0] = 1
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    return inbody, vmap

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    # ---------------------------
    vctx, vq, vas = stories # (B, Cs, C), (B, Q), (B, A)
    # Embed stories
    ectx = self.embed(vctx) # (B, Cs, C, E)
    eq = self.embed(vq) # (B, Q, E)
    # ---------------------------
    # Prepare rules and variable states
    rvctx, rvq, rva = self.vrules # (R, Ls, L), (R, Q), (R, A)
    rmctx, rmq, rma = self.mrules # (R, Ls, L), (R, Q), (R, A)
    erctx, erq, era = [self.embed(v) for v in self.vrules] # (R, Ls, L, E), (R, Q, E), (R, A, E)
    inbody, vmap = self.rulegen(self.vrules, [erctx, erq, era], self.temporal_enc) # (R, Ls, 2), (R, V)
    # inbody = np.zeros((9,2), dtype=np.float32)
    # inbody[0,0] = 1
    # inbody[2,0] = 1
    # inbody[3,0] = 1
    # vmap = np.array([[0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,]], dtype=np.float32)
    nrules_range = np.arange(len(self.rule_stories)) # (R,)
    ctxvgates = vmap[nrules_range[:, None, None], rvctx] # (R, Ls, L)
    qvgates, avgates = [vmap[nrules_range[:, None], v] for v in (rvq, rva)] # (R, Q), (R, A)
    eyeunk = self.eye * self.unkbias # (V, V)
    # vs_init = F.tile(eyeunk, (vq.shape[0], len(self.rule_stories), 1, 1)) # (B, R, V, V)
    vs_init = F.tile(eyeunk[0], (vq.shape[0], len(self.rule_stories), len(word2idx), 1)) # (B, R, V, V)
    vs = vs_init # (B, R, V, V)
    wordsrange = np.arange(len(word2idx)) # (V,)
    batchrange = np.arange(vctx.shape[0]) # (B,)
    # ---------------------------
    # Compute iterative updates on variables
    for _ in range(ITERATIONS):
      # ---------------------------
      # Unify query first assuming given query is ground
      qvvalues = vs[:, nrules_range[:, None], rvq, :] # (B, R, Q, V)
      qvvalues = F.softmax(qvvalues, -1) # (B, R, Q, V)
      qvvalues = qvvalues @ self.embed.W # (B, R, Q, E)
      qtoprove = qvgates[..., None]*qvvalues + (1-qvgates[..., None])*erq # (B, R, Q, E)
      qtoprove *= rmq[..., None] # (B, R, Q, E)
      qscores, qunis = self.unify(rvq[:, None, :], erq[:, None, ...], qtoprove[:, :, None, ...],
                                  vq[:, None, ...], eq[:, None, ...], self.temporal_enc) # (B, R, 1), (B, R, 1, Q, V)
      qunis = F.squeeze(qunis, 2) # (B, R, Q, V)
      # Update variable states with new unifications
      # np.isin flattens second argument, so we need for loop
      mask = np.vstack([np.isin(wordsrange, _rvq, invert=True) for _rvq in rvq]) # (R, V)
      vs *= mask[..., None] # (B, R, V, V)
      vs = F.scatter_add(vs, (batchrange[:, None, None], nrules_range[None, :, None], rvq), qunis) # (B, R, V, V)
      # ---------------------------
      # Unify body with updated variable state
      bodyvvalues = vs[:, nrules_range[:, None, None], rvctx, :] # (B, R, Ls, L, V)
      bodyvvalues = F.softmax(bodyvvalues, -1) # (B, R, Ls, L, V)
      bodyvvalues = bodyvvalues @ self.embed.W # (B, R, Ls, L, E)
      bodytoprove = ctxvgates[..., None]*bodyvvalues + (1-ctxvgates[..., None])*erctx # (B, R, Ls, L, E)
      bodytoprove *= rmctx[..., None] # (B, R, Ls, L, E)
      bscores, bunis = self.unify(rvctx, erctx, bodytoprove, vctx, ectx, self.temporal_enc) # (B, R, Ls), (B, R, Ls, L, V)
      # ---------------------------
      # Weighted update of variable states after body unification
      body_weights = inbody[..., 0] * bscores # (B, R, Ls)
      weighted_bunis = body_weights[..., None, None] * bunis # (B, R, Ls, L, V)
      normalisations = self.eye[rvctx] # (R, Ls, L, V)
      normalisations = F.einsum("ijk,jklm->ijm", bscores, normalisations) # (B, R, V)
      normalisations = normalisations[batchrange[:, None, None, None], nrules_range[None, :, None, None], rvctx[None, ...]] # (B, R, Ls, L)
      weighted_bunis /= normalisations[..., None] # (B, R, Ls, L, V)
      # np.isin flattens second argument, so we need for loop
      mask = np.vstack([np.isin(wordsrange, _rvctx, invert=True) for _rvctx in rvctx]) # (R, V)
      mask[:,0] = True # Padding is not updated
      vs *= mask[..., None] # (B, R, V, V)
      vs = F.scatter_add(vs, (batchrange[:, None, None, None], nrules_range[None, :, None, None], rvctx), weighted_bunis) # (B, R, V, V)
    # ---------------------------
    # Compute overall rule scores
    qscores = F.sigmoid(F.squeeze(qscores, -1)) # (B, R)
    bscores = F.sigmoid(bscores) # (B, R, Ls)
    # Computed negated scores: n(1-b) + (1-n)b
    isneg = inbody[..., 1] # (R, Ls)
    nbscores = isneg*(1-bscores) + (1-isneg)*bscores # (B, R, Ls)
    # Compute final scores for body premises: in*nb+(1-in)*1
    isin = inbody[..., 0] # (R, Ls)
    fbscores = isin*nbscores + (1-isin) # (B, R, Ls)
    # Final score for rule following AND semantics
    fbscores = F.cumprod(fbscores, -1)[..., -1] # (B, R)
    rscores = qscores * fbscores # (B, R)
    # ---------------------------
    # Compute answers based on variable and rule scores
    agrounds = eyeunk[rva] # (R, A, V)
    avvalues = vs[:, nrules_range[:, None], rva, :] # (B, R, A, V)
    predictions = avgates[None, ..., None]*avvalues + (1-avgates[..., None])*agrounds # (B, R, A, V)
    noans = F.tile(eyeunk[0], predictions.shape[:-1] + (1,)) # (B, R, A, V)
    answers = rscores[..., None, None]*predictions + (1-rscores[..., None, None])*noans # (B, R, A, V)
    # ---------------------------
    # Compute rule attentions
    num_rules = rvq.shape[0] # R
    if num_rules == 1:
      return answers[:, 0, 0, :], F.sum(vmap) * 0.01 # (B, V), ()
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

train_iter = C.iterators.SerialIterator(enc_stories, 32)
def converter(batch_stories, _):
  """Coverts given batch to expected format for Classifier."""
  vctx, vq, vas = vectorise_stories(batch_stories) # (B, Cs, C), (B, Q), (B, A)
  return (vctx, vq, vas), vas[:, 0] # (B,)
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
# trainer = T.Trainer(updater, T.triggers.EarlyStoppingTrigger())
trainer = T.Trainer(updater, (50, 'epoch'))

# Trainer extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 32, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_best.npz'), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
# trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=ARGS.name+'_log.json'))
# trainer.extend(T.extensions.LogReport(trigger=(1, 'iteration'), log_name=ARGS.name+'_log.json'))
trainer.extend(T.extensions.FailOnNonNumber())
trainer.extend(T.extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
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

# Hit the train button
try:
  trainer.run()
except KeyboardInterrupt:
  pass

# Print final rules
print([decode_story(rs) for rs in model.rule_stories])
print(model.vrules)
print(model.gen_rules())
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
  import ipdb; ipdb.set_trace()
  answer = model(converter([val_story], None)[0])
