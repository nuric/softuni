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
      self.match_linear = L.Linear(3*EMBED, EMBED)
      self.match_score = L.Linear(EMBED, 1)
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
    cbows += self.temporal_enc[:cbows.shape[0],:] # (len(candidates), E)
    # raw_scores = cbows @ pbow.T # (len(candidates), 1)
    pcbows = F.concat([pbow, cbows, pbow*cbows]) # (len(candidates), 3*E)
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
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.embed = L.EmbedID(len(word2idx), EMBED)
      # self.rulegen = RuleGen()
      self.var_linear = L.Linear(EMBED, EMBED)
      self.unify = Unify()
      self.unkbias = C.Parameter(4.0, shape=(1,), name="unkbias")

  def infer(self, story, rule_stories):
    """Given story and rules predict answers."""
    # Encode story
    embedded_ctx = sequence_embed(story['context'], self.embed) # [(s1len, E), (s2len, E), ...]
    embedded_q = sequence_embed([story['query']], self.embed)[0] # (qlen, E)
    # ---------------------------
    # Iterative theorem proving
    rules = list()
    unk = F.pad(self.unkbias, (0, len(word2idx)-1), 'constant', constant_values=0.0)
    for rs in rule_stories:
      # Encode rule story
      enc_rule = {'answers': sequence_embed([rs['answers']], self.embed)[0],
                  'query': sequence_embed([rs['query']], self.embed)[0],
                  'context': sequence_embed(rs['context'], self.embed)}
      # r = self.rulegen(rs, enc_rule) # Differentiable rule generated from story
      r = {'story': rs, 'bodymap': np.array([[0.0,0.0], [1.0,0.0]], dtype=np.float32),
           'vmap': {1: np.array(0.0, dtype=np.float32), 2: np.array(0.0, dtype=np.float32), 3: np.array(0.0, dtype=np.float32), 4: np.array(0.0, dtype=np.float32), 5: np.array(0.0, dtype=np.float32), 6: np.array(1.0, dtype=np.float32), 7: np.array(0.0, dtype=np.float32), 8: np.array(1.0, dtype=np.float32), 9: np.array(0.0, dtype=np.float32), 10: np.array(0.0, dtype=np.float32)}}
      vs = {vidx: F.pad(self.unkbias, (vidx, len(word2idx)-1-vidx), 'constant', constant_values=0.0)
            for vidx in r['vmap'].keys()} # Init unknown for every variable
      rules.append((r, vs, enc_rule))
    # ---------------------------
    # Compute iterative updates on variables
    rscores = list() # final rule scores
    for rule, vs, enc_rule in rules:
      # Iterative proving
      rwords = [rule['story']['query']]+rule['story']['context'] # [(qlen,), (s1len,), ...]
      grounds = (enc_rule['query'],)+enc_rule['context'] # [(qlen, E), (s1len, E), ...]
      vgates = [F.expand_dims(F.hstack([rule['vmap'][widx] for widx in widxs]), 1)
                for widxs in rwords] # [(qlen, 1), (s1len, 1), ...]
      for _ in range(1):
        # ---------------------------
        # Unify query first assuming given query is ground
        qvvalues = F.vstack([F.softmax(vs[widx], 0) @ self.embed.W for widx in rwords[0]]) # (qlen, E)
        qvvalues = self.var_linear(qvvalues) # (qlen, E)
        qtoprove = vgates[0]*qvvalues + (1-vgates[0])*grounds[0] # (qlen, E)
        qscore, qunified = self.unify(qtoprove, [story['query']], (embedded_q,)) # (), (qlen, len(word2idx))
        # Update variable states with new unifications
        for pidx, widx in enumerate(rwords[0]):
          vs[widx] = qunified[pidx]
        # ---------------------------
        # Regather all variable values
        vvalues = [F.vstack([F.softmax(vs[widx], 0) @ self.embed.W for widx in widxs]) for widxs in rwords[1:]]
                  # [(s1len, E), ...]
        vvalues = [self.var_linear(vv) for vv in vvalues] # [(s1len, E), ...]
        # Merge ground with variable values
        bodytoprove = [vg*vv+(1-vg)*gr for vg, vv, gr in zip(vgates[1:], vvalues, grounds[1:])]
        # ---------------------------
        # Unifications give new variable values and a score for match
        scores, unifications = [qscore], [qunified]
        # Unify body conditions
        for btoprove in bodytoprove:
          score, unified = self.unify(btoprove, story['context'], embedded_ctx) # (), (snlen, len(word2idx))
          scores.append(score)
          unifications.append(unified)
        # ---------------------------
        # Update variables after unification
        words = np.concatenate(rwords) # (qlen+s1len+s2len+...,)
        weights = [F.repeat(scores[0], len(rwords[0]))] # [(qlen,)
        inbody = rule['bodymap'][:,0] # (len(body),)
        weights.extend([F.repeat(scores[i]*inbody[i], len(seq)) for i, seq in enumerate(rwords[1:])]) # [(qlen,), (s1len,), ...]
        weights = F.hstack(weights) # (qlen+s1len+s2len+...,)
        unifications = F.concat(unifications, 0) # (qlen+s1len+s2len+..., len(word2idx))
        # Weighted sum based on score
        normalisations = {widx:0.1 for widx in vs.keys()}
        for pidx, widx in enumerate(words):
          normalisations[widx] += weights[pidx]
        # Reset variable states
        for widx in vs.keys():
          vs[widx] = self.xp.zeros(len(word2idx)) # (len(word2idx),)
        # Update new variable states
        for pidx, widx in enumerate(words):
          vs[widx] += (weights[pidx] * unifications[pidx] / normalisations[widx])
      # ---------------------------
      # Compute overall score for rule
      # rule['bodymap'].shape == (len(body), 2) => inbody, isnegated
      prem_scores = F.sigmoid(F.hstack(scores)) # (1 + len(body),)
      qscore, bscores = prem_scores[0], prem_scores[1:] # (), (len(body),)
      # Computed negated scores: n(1-b) + (1-n)b
      isneg = rule['bodymap'][:,1] # (len(body),)
      nbscores = isneg*(1-bscores) + (1-isneg)*bscores # (len(body),)
      # Compute final scores for body premises: in*nb+(1-in)*1
      inbody = rule['bodymap'][:,0] # (len(body),)
      fbscores = inbody*nbscores+(1-inbody) # (len(body),)
      # Final score for rule following AND semantics
      rscore = qscore * F.cumprod(fbscores)[-1] # ()
      rscores.append(rscore)
    # ---------------------------
    # Weighted sum using rule scores to produce final result
    # *** Just single rule case for now***
    # Read head of rule with variable mapping
    rscore = rscores[0]
    rule, vs, _ = rules[0]
    # Get ground value
    eye = self.xp.eye(len(word2idx)) # (len(word2idx), len(word2idx))
    asground = eye[rule['story']['answers']] * self.unkbias # (len(answers), len(word2idx))
    # Get variable values
    asvar = F.vstack([vs[widx] for widx in rule['story']['answers']]) # (len(answers), len(word2idx))
    # Get maximum appearance of variable
    varinbody = [[widx in seq for seq in rule['story']['context']] for widx in rule['story']['answers']]
    varinbody = np.array(varinbody) # (len(answers), len(body))
    inbody = rule['bodymap'][:,0] # (len(body),)
    maxinbody = F.hstack([F.max(inbody[vb]) for vb in varinbody]) # (len(answers),)
    # Compute final value using variable gating values
    asvgates = F.hstack([rule['vmap'][widx] for widx in rule['story']['answers']]) # (len(answers),)
    asvgates *= maxinbody # (len(answers),)
    asvgates = F.expand_dims(asvgates, 1) # (len(answers), 1)
    prediction = asvgates*asvar + (1-asvgates)*asground # (len(answers), len(word2idx))
    # Compute final final value using rule score
    noans = F.tile(unk, (len(rule['story']['answers']), 1)) # (len(answers), len(word2idx))
    answer = rscore*prediction + (1-rscore)*noans # (len(answers), len(word2idx))
    return answer #, rscore

  def forward(self, stories):
    """Compute the forward inference pass for given stories."""
    preds = [self.infer(s, rule_repo)[0] for s in stories] # [(len(word2idx),), ...]
    return F.vstack(preds) # (len(stories), len(word2idx))

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
model = Infer()
cmodel = L.Classifier(model, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)

optimiser = C.optimizers.Adam().setup(cmodel)
# optimiser.add_hook(C.optimizer_hooks.WeightDecay(0.01))

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
answer, confidence = model.infer(enc_stories[1], rule_repo)
