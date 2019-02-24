"""bAbI run on neurolog."""
import argparse
import logging
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
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
ARGS = parser.parse_args()

# Debug
if ARGS.debug:
  logging.basicConfig(level=logging.DEBUG)

EMBED = 16

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
        ss.append({'context': cctx, 'query': q,
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
  es['context'] = [np.array([word2idx.setdefault(w, len(word2idx)) for w in tokenise(s)]) for s in story['context']]
  es['query'] = np.array([word2idx.setdefault(w, len(word2idx)) for w in tokenise(story['query'])])
  es['answers'] = np.array([word2idx.setdefault(w, len(word2idx)) for w in story['answers']])
  return es
enc_stories = list(map(encode_story, stories))
print("TRAIN VOCAB:", len(word2idx))
val_enc_stories = list(map(encode_story, val_stories))
print("VAL VOCAB:", len(word2idx))
print("ENC SAMPLE:", enc_stories[0])
idx2word = {v:k for k, v in word2idx.items()}

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

def pos_encode(exs):
  """Given sentences compute positional encoding."""
  # [(s1len, E), (s2len, E), ...]
  def get_weights(slen):
    return np.fromfunction(lambda j, k: 1 - (j + 1) / slen - (k + 1) / EMBED * (1 - 2 * (j + 1) / slen),
                           (slen, EMBED), dtype=np.float32)
  return [F.sum(e * get_weights(e.shape[0]), axis=0) for e in exs]

# ---------------------------

# Rule generating network
class RuleGen(C.Chain):
  """Takes an example story-> context, query, answer
  returns a probabilistic rule"""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.bodylinear = L.Linear(EMBED*3, 2)
      self.convolve_words = L.Convolution1D(EMBED, EMBED, 3, pad=1)
      self.isvariable_linear = L.Linear(EMBED+2, 1)

  def forward(self, story, embedded_story):
    """Given a story generate a probabilistic learnable rule."""
    # Encode sequences
    embedded_ctx = embedded_story['context'] # [(s1len, E), (s2len, E), ...]
    enc_ctx = pos_encode(embedded_ctx) # [(E,), (E,), ...]
    enc_ctx = F.concat([F.expand_dims(e, 0) for e in enc_ctx], axis=0) # (clen, E)
    embedded_q = embedded_story['query'] # (qlen, E)
    enc_query = pos_encode([embedded_q])[0] # (E,)
    embedded_as = embedded_story['answers'] # (alen, E)
    enc_answer = pos_encode([embedded_as])[0] # (E,)
    # ---------------------------
    # Whether a fact is in the body or not, negated or not, multi-label -> (clen, 2)
    r_answer = F.repeat(F.expand_dims(enc_answer, 0), len(story['context']), axis=0) # (clen, E)
    r_query = F.repeat(F.expand_dims(enc_query, 0), len(story['context']), axis=0) # (clen, E)
    r_ctx = F.concat([r_answer, r_query, enc_ctx], axis=1) # (clen, E*3)
    ctxinbody = self.bodylinear(r_ctx) # (clen, 2)
    ctxinbody = F.sigmoid(ctxinbody) # (clen, 2)
    # ---------------------------
    # Whether each word in story is a variable, binary class -> {wordid:sigmoid}
    words = np.concatenate([story['query']]+story['context']) # (qlen+s1len+s2len+...,)
    unique_idxs, inverse_idxs, unique_counts = np.unique(words, return_inverse=True,
                                                         return_counts=True)
    # unique_idxs[inverse_idxs] == words
    # len(inverse_idxs) == len(words)
    # len(unique_counts) == len(unique_idxs)
    # Compute contextuals by convolving each sentence
    cwords = [F.squeeze(self.convolve_words(F.expand_dims(s.T, 0)), 0).T
              for s in (embedded_q,)+embedded_ctx] # [(qlen,16), (s1len,16), (s2len,16), ...]
    allwords = F.concat(cwords, axis=0) # (qlen+s1len+s2len+..., 16)
    assert len(allwords) == len(words), "Convolved features do not match story len."
    # Add whether they appear more than once
    appeartwice = (unique_counts[inverse_idxs] > 1) # (qlen+s1len+s2len+...,)
    appeartwice = appeartwice.astype(np.float32).reshape(-1, 1) # (qlen+s1len+s2len+..., 1)
    appearanswer = np.array([w in story['answers'] for w in words], dtype=np.float32).reshape(-1, 1) # (qlen+s1len+s2len+..., 1)
    allwords = F.concat([allwords, appeartwice, appearanswer], axis=1) # (qlen+s1len+s2len+..., 18)
    wordvars = self.isvariable_linear(allwords) # (qlen+s1len+s2len+..., 1)
    wordvars = F.squeeze(wordvars, 1) # (qlen+s1len+s2len+...,)
    wordvars = F.sigmoid(wordvars) # (qlen+s1len+s2len+...,)
    # Merge word variable predictions
    iswordvar = {idx:1.0 for idx in unique_idxs}
    for pidx, widx in enumerate(words):
      iswordvar[widx] *= wordvars[pidx]
    # ---------------------------
    # Tells whether a context is in the body, negated or not
    # and whether a word is a variable or not
    return {'story': story, 'bodymap': ctxinbody, 'vmap': iswordvar}

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
      self.temporal_enc = C.Parameter(C.initializers.Normal(1.0), (20, EMBED), name="tempenc")

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
                   # len(candidates) [(plen, len(word2idx)), ...]
    unifications = F.stack(unifications, axis=0) # (len(candidates), plen, len(word2idx)
    # Weighted sum using scores
    weights = F.softmax(raw_scores, 0) # (len(candidates),)
    final_uni = F.einsum("i,ijk->jk", weights, unifications) # (plen, len(word2idx))
    final_score = F.max(raw_scores) # ()
    return final_score, final_uni

# ---------------------------

# Inference network
class Infer(C.Chain):
  """Takes a story, a set of rules and predicts answers."""
  def __init__(self, rule_stories):
    super().__init__()
    self.rule_stories = rule_stories
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED)
      # self.rulegen = RuleGen()
      self.var_linear = L.Linear(EMBED, EMBED)
      self.unify = Unify()
      self.unkbias = C.Parameter(4.0, shape=(1,), name="unkbias")

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    # ---------------------------
    # Prepare rules from rule stories
    rules = list()
    vs_init = self.xp.eye(len(word2idx), dtype=self.xp.float32) * self.unkbias # (len(word2idx), len(word2idx))
    for rs in self.rule_stories:
      # Encode rule story
      enc_rule = story_embed(rs, self.embed)
      # r = self.rulegen(rs, enc_rule) # Differentiable rule generated from story
      r = {'story': rs, 'bodymap': np.array([[0.0,0.0], [1.0,0.0]], dtype=np.float32),
           'vmap': np.array([0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)}
      vs = F.tile(vs_init, (len(stories), 1, 1)) # (B, len(word2idx), len(word2idx))
      rules.append([r, vs, enc_rule])
    # ---------------------------
    # Embed stories
    embedded_stories = [story_embed(s, self.embed) for s in stories] # (B, [{'answers': (alen, E), 'query': (qlen, E), 'context': [(s1len, E), ...]}, ...])
    batch_range = np.arange(len(stories)).reshape(-1, 1) # (B, 1)
    # ---------------------------
    # Compute iterative updates on variables
    rscores = list() # final rule scores
    wordsrange = np.arange(len(word2idx)) # (len(word2idx),)
    for comprule in rules:
      rule, vs, enc_rule = comprule
      # Iterative proving
      rwords = [rule['story']['query']]+rule['story']['context'] # [(qlen,), (s1len,), ...]
      slens = [len(s) for s in rwords] # [qlen, s1len, s2len, ...]
      concat_rwords = np.concatenate(rwords) # (qlen+s1len+s2len+...,)
      grounds = F.concat((enc_rule['query'],)+enc_rule['context'], 0) # (qlen+s1len+s2len+..., E)
      vgates = F.expand_dims(rule['vmap'][concat_rwords], 1) # (qlen+s1len+s2len+..., 1)
      for _ in range(1):
        # ---------------------------
        # Unify query first assuming given query is ground
        qvvalues = F.softmax(vs[:, rwords[0]], -1) # (B, qlen, len(word2idx))
        qvvalues = qvvalues @ self.embed.W # (B, qlen, E)
        qvvalues = self.var_linear(qvvalues, n_batch_axes=2) # (B, qlen, E)
        qlen = len(rwords[0])
        qtoprove = vgates[:qlen]*qvvalues + (1-vgates[:qlen])*grounds[:qlen] # (B, qlen, E)
        qscores, qunifieds = list(), list()
        for i, (s, es) in enumerate(zip(stories, embedded_stories)):
          qscore, qunified = self.unify(qtoprove[i], [s['query']], (es['query'],))
          qscores.append(qscore) # ()
          qunifieds.append(qunified) # (qlen, len(word2idx))
        qscores = F.vstack(qscores) # (B, 1)
        qunifieds = F.stack(qunifieds, 0) # (B, qlen, len(word2idx))
        # Update variable states with new unifications
        mask = np.isin(wordsrange, rwords[0], invert=True).astype(np.float32) # (len(word2idx),)
        vs *= mask[:, None] # (B, len(word2idx), len(word2idx))
        vs = F.scatter_add(vs, (batch_range, rwords[0]), qunifieds) # (len(word2idx), len(word2idx))
        # ---------------------------
        # Regather all variable values
        vvalues = F.softmax(vs[:, concat_rwords[qlen:]], -1) # (B, s1len+s2len+..., len(word2idx))
        vvalues = vvalues @ self.embed.W # (B, s1len+s2len+..., E)
        vvalues = self.var_linear(vvalues, n_batch_axes=2) # (B, s1len+s2len+..., E)
        # Merge ground with variable values
        bodytoprove = vgates[qlen:]*vvalues + (1-vgates[qlen:])*grounds[qlen:] # (B, s1len+s2len+..., E)
        # ---------------------------
        # Unifications give new variable values and a score for match
        scores, unifications = [qscores], [qunifieds] # [(B, 1)], [(B, qlen, len(word2idx)]
        # Unify body conditions
        bodiestoprove = F.split_axis(bodytoprove, np.cumsum(slens[1:-1]), 1) # [(B, s1len, E), (B, s2len, E), ...]
        for btoprove in bodiestoprove:
          bscores, bunifications = list(), list()
          for i, (s, es) in enumerate(zip(stories, embedded_stories)):
            score, unified = self.unify(btoprove[i], s['context'], es['context']) # (), (snlen, len(word2idx))
            bscores.append(score) # ()
            bunifications.append(unified) # (snlen, len(word2idx))
          scores.append(F.vstack(bscores)) # (B, 1)
          unifications.append(F.stack(bunifications, 0)) # (B, snlen, len(word2idx))
        # ---------------------------
        # Update variables after unification
        vscores = F.concat(scores, 1) # (B, 1+len(body))
        weights = np.repeat(np.arange(len(slens)), slens) # (qlen+s1len+...,)
        weights = vscores[:, weights] # (B, qlen+s1len+...)
        inbody = F.repeat(rule['bodymap'][:,0], tuple(slens[1:])) # (s1len+s2len+...,)
        inbody = F.pad(inbody, (slens[0], 0), 'constant', constant_values=1.0) # (qlen+s1len+...,)
        weights *= inbody # (B, qlen+s1len+...)
        normalisations = F.scatter_add(self.xp.full((len(stories), len(word2idx)), 0.001, dtype=self.xp.float32),
                                       (batch_range, concat_rwords), weights) # (B, len(word2idx))
        unifications = F.concat(unifications, 1) # (B, qlen+s1len+s2len+..., len(word2idx))
        # Weighted sum based on score
        unifications *= weights[..., None] # (B, qlen+s1len+s2len+..., len(word2idx))
        unifications = F.scatter_add(self.xp.zeros((len(stories), len(word2idx), len(word2idx)), dtype=self.xp.float32),
                                     (batch_range, concat_rwords), unifications) # (B, len(word2idx), len(word2idx))
        comprule[1] = unifications / normalisations[..., None]
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
    # *** Just single rule case for now***
    # Read head of rule with variable mapping
    rscore = rscores[0]
    rule, vs, _ = rules[0]
    aswords = rule['story']['answers'] # (len(answers,)
    # Get ground value
    eye = self.xp.eye(len(word2idx)) # (len(word2idx), len(word2idx))
    asground = eye[aswords] * self.unkbias # (len(answers), len(word2idx))
    # Get variable values
    asvar = vs[:, aswords] # (B, len(answers), len(word2idx))
    # Get maximum appearance of variable
    varinbody = [[widx in seq for seq in rule['story']['context']] for widx in aswords]
    varinbody = np.array(varinbody) # (len(answers), len(body))
    inbody = rule['bodymap'][:,0] # (len(body),)
    maxinbody = varinbody * inbody # (len(answers), len(body))
    maxinbody = F.max(maxinbody, -1) # (len(answers),)
    # Compute final value using variable gating values
    asvgates = rule['vmap'][aswords] # (len(answers),)
    asvgates *= maxinbody # (len(answers),)
    prediction = asvgates[:,None]*asvar + (1-asvgates[:,None])*asground # (B, len(answers), len(word2idx))
    # Compute final final value using rule score
    noans = F.tile(vs_init[0], (len(stories), len(aswords), 1)) # (B, len(answers), len(word2idx))
    answers = rscore[:, None, None]*prediction + (1-rscore[:, None, None])*noans # (B, len(answers), len(word2idx))
    # Assume single word answer for now
    return answers[:, 0, :]

# ---------------------------

# Stories to generate rules from
rule_repo = [enc_stories[0]]
answers = set()
# training = list()
# init_tsize = 9
# for es in enc_stories[1:]:
  # if es['answers'][0] not in answers:
    # training.append(es)
    # answers.add(es['answers'][0])
  # if len(training) == init_tsize:
    # break

# ---------------------------

# Setup model
model = Infer(rule_repo)
cmodel = L.Classifier(model, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)

optimiser = C.optimizers.Adam().setup(cmodel)
optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.001))

train_iter = C.iterators.SerialIterator(enc_stories, 32)
def converter(stories, _):
  """Coverts given batch to expected format for Classifier."""
  return stories, np.array([s['answers'][0] for s in stories])
updater = T.StandardUpdater(train_iter, optimiser, converter=converter, device=-1)
trainer = T.Trainer(updater, (50, 'epoch'))

# Trainer extensions
val_iter = C.iterators.SerialIterator(val_enc_stories, 32, repeat=False, shuffle=False)
trainer.extend(T.extensions.Evaluator(val_iter, cmodel, converter=converter, device=-1))
# trainer.extend(T.extensions.snapshot(), trigger=T.triggers.MinValueTrigger('validation/main/loss'))
trainer.extend(T.extensions.LogReport(trigger=(1,'iteration')))
trainer.extend(T.extensions.FailOnNonNumber())
trainer.extend(T.extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
# trainer.extend(T.extensions.ProgressBar(update_interval=10))
trainer.extend(T.extensions.PlotReport(['main/loss', 'validation/main/loss'], 'iteration', marker=None, file_name='loss.pdf'))
trainer.extend(T.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'iteration', marker=None, file_name='accuracy.pdf'))

try:
  trainer.run()
except KeyboardInterrupt:
  pass
import ipdb; ipdb.set_trace()
answer = model([enc_stories[1], enc_stories[20]])
