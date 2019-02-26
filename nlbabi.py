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
matplotlib.use('pdf')


# Disable scientific printing
np.set_printoptions(suppress=True, precision=3)

# Arguments
parser = argparse.ArgumentParser(description="Run NeuroLog on bAbI tasks.")
parser.add_argument("task", help="File that contains task train.")
parser.add_argument("validation", help="File that contains task validation.")
parser.add_argument("--name", default="nlbabi", help="Name prefix for saving files etc.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
ARGS = parser.parse_args()

# Debug
if ARGS.debug:
  logging.basicConfig(level=logging.DEBUG)
  C.set_debug(True)

EMBED = 16
MAX_HIST = 25
REPO_SIZE = 2
ITERATIONS = 3

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
val_stories = load_task(ARGS.validation)
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

# ---------------------------

# Rule generating network
class RuleGen(C.Chain):
  """Takes an example story-> context, query, answer
  returns a probabilistic rule"""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.body_linear = L.Linear(5*EMBED, 2*EMBED)
      self.body_score = L.Linear(2*EMBED, 2)
      self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.isvariable_linear = L.Linear(EMBED+1, EMBED)
      self.isvariable_score = L.Linear(EMBED, 1)

  def contextual_convolve(self, vxs, exs):
    """Given vectorised and encoded sentences convolve over last dimension."""
    # (R, ..., S), (R, ..., S, E)
    mask = (vxs != 0.0) # (R, ..., S)
    toconvolve = exs # Assuming we have (R, S), (R, S, E)
    if len(vxs.shape) > 2:
      # We need to pad between sentences
      padding = self.xp.zeros(exs.shape[:-2]+(1,exs.shape[-1]), dtype=exs.dtype) # (R, ..., 1, E)
      padded = F.concat([exs, padding], -2) # (R, ..., S+1, E)
      toconvolve = F.reshape(padded, (exs.shape[0], -1, exs.shape[-1])) # (R, *S+1, E)
    permuted = F.transpose(toconvolve, (0, 2, 1)) # (R, E, S)
    contextual = self.convolve_words(permuted) # (R, E, S)
    contextual = F.transpose(contextual, (0, 2, 1)) # (R, S, E)
    if len(vxs.shape) > 2:
      contextual = F.reshape(contextual, padded.shape) # (R, ..., S+1, E)
      contextual = contextual[..., :-1, :] # (R, ..., S, E)
    contextual *= mask[..., None] # (R, ..., S, E)
    return contextual

  def forward(self, vectorised_rules, embedded_rules):
    """Given a story generate a probabilistic learnable rule."""
    # vectorised_rules = [(R, Cs, C), (R, Q), (R, A)]
    # embedded_rules = [(R, Cs, C, E), (R, Q, E), (R, A, E)]
    vctx, vq, va = vectorised_rules
    ectx, eq, ea = embedded_rules
    # Encode sequences
    enc_ctx = pos_encode(vctx, ectx) # (R, Cs, E)
    enc_query = pos_encode(vq, eq) # (R, E)
    enc_answer = pos_encode(va, ea) # (R, E)
    # ---------------------------
    # Whether a fact is in the body or not, negated or not, multi-label -> (clen, 2)
    clen = enc_ctx.shape[1] # Cs
    r_answer = F.repeat(enc_answer[:, None, :], clen, 1) # (R, Cs, E)
    r_query = F.repeat(enc_query[:, None, :], clen, 1) # (R, Cs, E)
    r_ctx = F.concat([r_answer, r_query, enc_ctx, r_answer*enc_ctx, r_query*enc_ctx], -1) # (R, Cs, 5*E)
    inbody = self.body_linear(r_ctx, n_batch_axes=2) # (R, Cs, 2*E)
    inbody = F.tanh(inbody) # (R, Cs, 2*E)
    inbody = self.body_score(inbody, n_batch_axes=2) # (R, Cs, 2)
    inbody = F.sigmoid(inbody) # (R, Cs, 2)
    bodymask = (vctx != 0.0) # (R, Cs, S)
    bodymask = np.any(bodymask, -1) # (R, Cs)
    inbody *= bodymask[..., None] # (R, Cs, 2)
    # ---------------------------
    # Whether each word in story is a variable, binary class -> {wordid:sigmoid}
    num_rules = vctx.shape[0] # R
    words = np.reshape(vctx, (num_rules, -1)) # (R, Cs*C)
    words = np.concatenate([vq, words], -1) # (R, Q+Cs*C)
    # Compute contextuals by convolving each sentence
    contextual_q = self.contextual_convolve(vq, eq) # (R, Q, E)
    contextual_ctx = self.contextual_convolve(vctx, ectx) # (R, Cs, C, E)
    flat_cctx = F.reshape(contextual_ctx, (num_rules, -1, ectx.shape[-1])) # (R, Cs * C, E)
    cwords = F.concat([contextual_q, flat_cctx], 1) # (R, Q+Cs*C, E)
    # Add whether they appear in the answer as a featurec
    appearanswer = np.isin(words, va).astype(np.float32) # (R, Q+Cs*C)
    allwords = F.concat([cwords, appearanswer[..., None]], -1) # (R, Q+Cs*C, E+1)
    wordvars = self.isvariable_linear(allwords, n_batch_axes=2) # (R, Q+Cs*C, E)
    wordvars = F.tanh(wordvars) # (R, Q+Cs*C, E)
    wordvars = self.isvariable_score(wordvars, n_batch_axes=2) # (R, Q+Cs*C, 1)
    wordvars = F.squeeze(wordvars, -1) # (R, Q+Cs*C)
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
    return {'bodymap': inbody, 'vmap': iswordvar}

# ---------------------------

# Unification network
class Unify(C.Chain):
  """Semantic unification on two sentences with variables."""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.match_linear = L.Linear(6*EMBED, 3*EMBED)
      self.match_score = L.Linear(3*EMBED, 1)
      self.temporal_enc = C.Parameter(C.initializers.Normal(1.0), (MAX_HIST, EMBED), name="tempenc")

  def forward(self, toprove, candidates, embedded_candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (plen, E)
    # candidates = [(s1len,), (s2len,), ...]
    # embedded_candidates = [(s1len, E), (s2len, E), ...]
    assert len(candidates) == len(embedded_candidates), "Candidate lengths differ."
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    cwords = [F.squeeze(self.convolve_words(F.expand_dims(s.T, 0)), 0).T
              for s in (toprove,)+embedded_candidates] # [(plen,16), (s1len,16), (s2len,16]
    # Compute similarity between every candidate
    ctp, *ccandids = cwords # (plen,16), [(s1len,16), (s2len,16), ...]
    raw_sims = [ctp @ c.T for c in ccandids] # [(plen,s1len), (plen,s2len), ...]
    # ---------------------------
    # Calculate score for each candidate
    # Compute bag of words
    pbow, *cbows = [F.sum(s, axis=0) for s in cwords] # (E,) [(E,), ...]
    # pbow = pos_encode([toprove])[0] # (E,)
    # cbows = pos_encode(embedded_candidates) # [(E,), (E,)]
    pbow = F.repeat(F.expand_dims(pbow, 0), len(candidates), axis=0) # (len(candidates), E)
    cbows = F.vstack(cbows) # (len(candidates), E)
    tbows = self.temporal_enc[:cbows.shape[0], :] # (len(candidates), E)
    pcbows = F.concat([pbow, cbows, pbow*cbows, tbows, tbows+cbows, tbows*cbows]) # (len(candidates), 6*E)
    raw_scores = self.match_linear(pcbows) # (len(candidates, E)
    raw_scores = F.tanh(raw_scores) # (len(candidates), E)
    raw_scores = self.match_score(raw_scores) # (len(candidates, 1)
    raw_scores = F.squeeze(raw_scores, 1) # (len(candidates),)
    # ---------------------------
    # Calculate attended unified word representations for toprove
    eye = self.xp.eye(len(word2idx))
    unifications = [simvals @ eye[candidxs]
                    for simvals, candidxs in zip(raw_sims, candidates)]
                   # len(candidates) [(plen, V), ...]
    unifications = F.stack(unifications, axis=0) # (len(candidates), plen, V)
    # Weighted sum using scores
    weights = F.softmax(raw_scores, 0) # (len(candidates),)
    final_uni = F.einsum("i,ijk->jk", weights, unifications) # (plen, V)
    final_score = F.max(raw_scores) # ()
    return final_score, final_uni

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
      self.rulegen = RuleGen()
      self.var_linear = L.Linear(EMBED, EMBED)
      self.unify = Unify()
      self.unkbias = C.Parameter(4.0, shape=(1,), name="unkbias")
      self.weye = self.xp.eye(len(word2idx), dtype=self.xp.float32) * self.unkbias # (V, V)
    # Setup rule repo
    self.vrules = vectorise_stories(rule_stories) # (R, Cs, C), (R, Q), (R, A)
    self.erules = tuple([self.embed(v) for v in self.vrules]) # (R, Cs, C, E), (R, Q, E), (R, A, E)
    self.genrules = self.rulegen(self.vrules, self.erules) # {'bodymap': (R, Cs, 2), 'vmap': (R, V)}

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    # ---------------------------
    vctx, vq, vas = stories # (B, Cs, C), (B, Q), (B, A)
    import ipdb; ipdb.set_trace()
    print("HERE")
    # Embed stories
    ectx = self.embed(vctx) # (B, Cs, C, E)
    eq = self.embed(vq) # (B, Q, E)
    # ---------------------------
    # Compute iterative updates on variables
    rscores = list() # final rule scores
    wordsrange = np.arange(len(word2idx)) # (V,)
    for comprule in rules:
      rule, vs, enc_rule = comprule
      # Iterative proving
      rwords = [rule['story']['query']]+rule['story']['context'] # [(qlen,), (s1len,), ...]
      slens = [len(s) for s in rwords] # [qlen, s1len, s2len, ...]
      concat_rwords = np.concatenate(rwords) # (qlen+s1len+s2len+...,)
      grounds = F.concat((enc_rule['query'],)+enc_rule['context'], 0) # (qlen+s1len+s2len+..., E)
      vgates = F.expand_dims(rule['vmap'][concat_rwords], 1) # (qlen+s1len+s2len+..., 1)
      for _ in range(ITERATIONS):
        # ---------------------------
        # Unify query first assuming given query is ground
        qvvalues = F.softmax(vs[:, rwords[0]], -1) # (B, qlen, V)
        qvvalues = qvvalues @ self.embed.W # (B, qlen, E)
        qvvalues = self.var_linear(qvvalues, n_batch_axes=2) # (B, qlen, E)
        qlen = len(rwords[0])
        qtoprove = vgates[:qlen]*qvvalues + (1-vgates[:qlen])*grounds[:qlen] # (B, qlen, E)
        qscores, qunifieds = list(), list()
        for i, (s, es) in enumerate(zip(stories, embedded_stories)):
          qscore, qunified = self.unify(qtoprove[i], [s['query']], (es['query'],))
          qscores.append(qscore) # ()
          qunifieds.append(qunified) # (qlen, V)
        qscores = F.vstack(qscores) # (B, 1)
        qunifieds = F.stack(qunifieds, 0) # (B, qlen, V)
        # Update variable states with new unifications
        mask = np.isin(wordsrange, rwords[0], invert=True).astype(np.float32) # (V,)
        vs *= mask[:, None] # (B, V, V)
        vs = F.scatter_add(vs, (batch_range, rwords[0]), qunifieds) # (V, V)
        # ---------------------------
        # Regather all variable values
        vvalues = F.softmax(vs[:, concat_rwords[qlen:]], -1) # (B, s1len+s2len+..., V)
        vvalues = vvalues @ self.embed.W # (B, s1len+s2len+..., E)
        vvalues = self.var_linear(vvalues, n_batch_axes=2) # (B, s1len+s2len+..., E)
        # Merge ground with variable values
        bodytoprove = vgates[qlen:]*vvalues + (1-vgates[qlen:])*grounds[qlen:] # (B, s1len+s2len+..., E)
        # ---------------------------
        # Unifications give new variable values and a score for match
        scores, unifications = [qscores], [qunifieds] # [(B, 1)], [(B, qlen, V)]
        # Unify body conditions
        bodiestoprove = F.split_axis(bodytoprove, np.cumsum(slens[1:-1]), 1) # [(B, s1len, E), (B, s2len, E), ...]
        for btoprove in bodiestoprove:
          bscores, bunifications = list(), list()
          for i, (s, es) in enumerate(zip(stories, embedded_stories)):
            score, unified = self.unify(btoprove[i], s['context'], es['context']) # (), (snlen, V)
            bscores.append(score) # ()
            bunifications.append(unified) # (snlen, V)
          scores.append(F.vstack(bscores)) # (B, 1)
          unifications.append(F.stack(bunifications, 0)) # (B, snlen, V)
        # ---------------------------
        # Update variables after unification
        vscores = F.concat(scores, 1) # (B, 1+len(body))
        weights = np.repeat(np.arange(len(slens)), slens) # (qlen+s1len+...,)
        weights = vscores[:, weights] # (B, qlen+s1len+...)
        inbody = F.repeat(rule['bodymap'][:,0], tuple(slens[1:])) # (s1len+s2len+...,)
        inbody = F.pad(inbody, (slens[0], 0), 'constant', constant_values=1.0) # (qlen+s1len+...,)
        weights *= inbody # (B, qlen+s1len+...)
        normalisations = F.scatter_add(self.xp.full((len(stories), len(word2idx)), 0.001, dtype=self.xp.float32),
                                       (batch_range, concat_rwords), weights) # (B, V)
        unifications = F.concat(unifications, 1) # (B, qlen+s1len+s2len+..., V)
        # Weighted sum based on score
        unifications *= weights[..., None] # (B, qlen+s1len+s2len+..., V)
        unifications = F.scatter_add(self.xp.zeros((len(stories), len(word2idx), len(word2idx)), dtype=self.xp.float32),
                                     (batch_range, concat_rwords), unifications) # (B, V, V)
        vs = unifications / normalisations[..., None]
        comprule[1] = vs
      # ---------------------------
      # Compute overall score for rule
      # rule['bodymap'].shape == (len(body), 2) => inbody, isnegated
      prem_scores = F.sigmoid(vscores) # (B, 1 + len(body))
      qscore, bscores = prem_scores[:, 0], prem_scores[:, 1:] # (B,), (B, len(body))
      # Computed negated scores: n(1-b) + (1-n)b
      isneg = rule['bodymap'][:,1] # (len(body),)
      nbscores = isneg*(1-bscores) + (1-isneg)*bscores # (B, len(body))
      # Compute final scores for body premises: in*nb+(1-in)*1
      inbody = rule['bodymap'][:,0] # (len(body),)
      fbscores = inbody*nbscores+(1-inbody) # (B, len(body))
      # Final score for rule following AND semantics
      rscore = qscore * F.cumprod(fbscores, -1)[:, -1] # (B,)
      rscores.append(rscore)
    # ---------------------------
    # Weighted sum using rule scores to produce final result
    allanswers = list()
    max_alen = max([len(r[0]['story']['answers']) for r in rules])
    for rscore, (rule, vs, _) in zip(rscores, rules):
      # Read head of rule with variable mapping
      aswords = rule['story']['answers'] # (len(answers),)
      # Get ground value
      asground = self.weye[aswords] # (len(answers), V)
      # Get variable values
      asvar = vs[:, aswords] # (B, len(answers), V)
      # Get maximum appearance of variable
      varinbody = [[widx in seq for seq in rule['story']['context']] for widx in aswords]
      varinbody = np.array(varinbody) # (len(answers), len(body))
      inbody = rule['bodymap'][:,0] # (len(body),)
      maxinbody = varinbody * inbody # (len(answers), len(body))
      maxinbody = F.max(maxinbody, -1) # (len(answers),)
      # Compute final value using variable gating values
      asvgates = rule['vmap'][aswords] # (len(answers),)
      asvgates *= maxinbody # (len(answers),)
      prediction = asvgates[:,None]*asvar + (1-asvgates[:,None])*asground # (B, len(answers), V)
      # Compute final final value using rule score
      noans = F.tile(self.weye[0], (len(stories), len(aswords), 1)) # (B, len(answers), V)
      answers = rscore[:, None, None]*prediction + (1-rscore[:, None, None])*noans # (B, len(answers), V)
      # Shorcut for single rule cases
      if len(rules) == 1:
        return answers[:, 0, :]
      if len(aswords) != max_alen:
        # Pad answers to match shapes
        answers = F.pad(answers, (0, (0, max_alen-len(aswords)), 0), mode='constant', constant_values=0.0)
        import ipdb; ipdb.set_trace()
      allanswers.append(answers) # (B, max_alen, V)
    # Aggregate answers based on rule scores
    rscores = F.vstack(rscores) # (R, B)
    rscores = F.softmax(rscores, 0) # (R, B)
    allanswers = F.stack(allanswers, 0) # (R, B, max_alen, V)
    final_answers = F.einsum("ij,ijkl->jkl", rscores, allanswers) # (B, max_alen, V)
    # **Assume one answer for now**
    return final_answers[:, 0, :]

# ---------------------------

# Stories to generate rules from
answers = set()
rule_repo = list()
np.random.shuffle(enc_stories)
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
cmodel = L.Classifier(model, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)

optimiser = C.optimizers.Adam().setup(cmodel)
optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))

train_iter = C.iterators.SerialIterator(enc_stories, 7)
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
trainer.extend(T.extensions.snapshot(filename=ARGS.name+'_latest.npz'), trigger=(1, 'epoch'))
trainer.extend(T.extensions.LogReport(log_name=ARGS.name+'_log.json'))
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

if ARGS.debug:
  import ipdb; ipdb.set_trace()
  answers = model([enc_stories[1], enc_stories[20]])
  print("INIT ANSWERS:", answers)

# Hit the train button
try:
  trainer.run()
except KeyboardInterrupt:
  pass

# Print final rules
for r, _, _  in model.gen_rules():
  print("RSTORY:", decode_story(r['story']))
  print(r['bodymap'])
  print(r['vmap'])
# Extra inspection if we are debugging
if ARGS.debug:
  import ipdb; ipdb.set_trace()
  answers = model([enc_stories[1], enc_stories[20]])
  print("FINAL ANSWERS:", answers)
