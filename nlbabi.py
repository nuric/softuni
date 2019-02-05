"""bAbI run on neurolog."""
import argparse
import logging
import time
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F


# Disable scientific printing
np.set_printoptions(suppress=True, precision=3)

# Arguments
parser = argparse.ArgumentParser(description="Run NeuroLog on bAbI tasks.")
parser.add_argument("task", help="File that contains task.")
parser.add_argument("vocab", help="File contains word vectors.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
ARGS = parser.parse_args()

# Debug
if ARGS.debug:
  logging.basicConfig(level=logging.DEBUG)

# ---------------------------

# Load in task
stories = []
with open(ARGS.task) as f:
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
      stories.append({'context': cctx, 'query': q,
                      'answers': a.split(',')})
    else:
      # Just a statement
      context.append(sl)
    prev_id = sid
print("TOTAL:", len(stories), "stories")
print("SAMPLE:", stories[0])

# ---------------------------

# Load word vectors
wordvecs = [np.zeros(300, dtype=np.float32)]
word2idx = {'unk':0}
idx2word = {0:'unk'}
with open(ARGS.vocab) as f:
  for i, l in enumerate(f):
    word, *vec = l.split(' ')
    wordvecs.append(np.array([float(n) for n in vec], dtype=np.float32))
    word2idx[word], idx2word[i+1] = i+1, word
wordvecs = np.array(wordvecs, dtype=np.float32)
print("VOCAB:", wordvecs.shape)

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

# Encode stories
def encode_story(story):
  """Convert given story into word vector indices."""
  es = dict()
  es['context'] = [np.array([word2idx[w] for w in tokenise(s)]) for s in story['context']]
  es['query'] = np.array([word2idx[w] for w in tokenise(story['query'])])
  es['answers'] = np.array([word2idx[w] for w in story['answers']])
  return es
enc_stories = C.datasets.TransformDataset(stories, encode_story)
print(enc_stories[0])

# ---------------------------

# Utility functions for neural networks
def sequence_embed(seqs):
  """Embed sequences of integer ids to word vectors."""
  x_len = [len(x) for x in seqs]
  x_section = np.cumsum(x_len[:-1])
  ex = F.embed_id(F.concat(seqs, axis=0), wordvecs)
  exs = F.split_axis(ex, x_section, 0)
  return exs

def bow_encode(exs):
  """Given sentences compute is bag-of-words representation."""
  return [F.sum(e, axis=0) for e in exs]

# ---------------------------

# Rule generating network
class RuleGen(C.Chain):
  """Takes an example story-> context, query, answer
  returns a probabilistic rule"""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.bodylinear = L.Linear(900, 2)
      self.convolve_words = L.Convolution1D(300, 16, 3, pad=1)
      self.isvariable_linear = L.Linear(17, 1)

  def forward(self, story):
    """Given a story generate a probabilistic learnable rule."""
    # Encode sequences
    embedded_ctx = sequence_embed(story['context']) # [(s1len, 300), (s2len, 300), ...]
    enc_ctx = bow_encode(embedded_ctx) # [(300,), (300,), ...]
    enc_ctx = F.concat([F.expand_dims(e, 0) for e in enc_ctx], axis=0) # (clen, 300)
    embedded_q = sequence_embed([story['query']])[0] # (qlen, 300)
    enc_query = bow_encode([embedded_q])[0] # (300,)
    embedded_as = sequence_embed([story['answers']])[0] # (alen, 300)
    enc_answer = bow_encode([embedded_as])[0] # (300,)
    # ---------------------------
    # Whether a fact is in the body or not, negated or not, multi-label -> (clen, 2)
    r_answer = F.repeat(F.expand_dims(enc_answer, 0), len(story['context']), axis=0) # (clen, 300)
    r_query = F.repeat(F.expand_dims(enc_query, 0), len(story['context']), axis=0) # (clen, 300)
    r_ctx = F.concat([r_answer, r_query, enc_ctx], axis=1) # (clen, 300*3)
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
    allwords = F.concat([allwords, appeartwice], axis=1) # (qlen+s1len+s2len+..., 17)
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
      self.linear = L.Linear(300, 2)
      self.convolve_words = L.Convolution1D(300, 16, 3, pad=1)
      self.match_rnn = L.NStepGRU(1, 16, 16, 0.1)
      self.match_linear = L.Linear(16, 1)

  def forward(self, toprove, candidates, embedded_candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (plen, 300)
    # candidates = [(s1len,), (s2len,), ...]
    # embedded_candidates = [(s1len, 300), (s2len, 300), ...]
    assert len(candidates) == len(embedded_candidates), "Candidate lengths differ."
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    cwords = [F.squeeze(self.convolve_words(F.expand_dims(s.T, 0)), 0).T
              for s in (toprove,)+embedded_candidates] # [(plen,16), (s1len,16), (s2len,16]
    # Compute similarity between every candidate
    ctp, *ccandids = cwords # (plen,16), [(s1len,16), (s2len,16), ...]
    raw_sims = [ctp @ c.T for c in ccandids] # [(plen,s1len), (plen,s2len), ...]
    # raw_sims = [toprove @ c.T for c in embedded_candidates] # [(plen,s1len), (plen,s2len), ...]
    # ---------------------------
    # Calculate score for each candidate
    # Compute bag of words
    pbow, *cbows = [F.sum(s, axis=0, keepdims=True) for s in cwords] # [(1,16), (1,16)]
    pbow = F.expand_dims(pbow, 0) # (1,1,16) (layers, batchsize, 16)
    cbows = F.concat(cbows, axis=0) # (len(candidates), 16)
    _, raw_scores = self.match_rnn(pbow, [cbows]) # _, [(len(candidates), 16)]
    raw_scores = self.match_linear(raw_scores[0]) # (len(candidates), 1)
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
      self.rulegen = RuleGen()
      self.unify = Unify()

  def forward(self, story, rule_stories):
    """Given story and rules predict answers."""
    # Encode story
    embedded_ctx = sequence_embed(story['context']) # [(s1len, 300), (s2len, 300), ...]
    enc_ctx = bow_encode(embedded_ctx) # [(300,), (300,), ...]
    enc_ctx = F.vstack(enc_ctx) # (clen, 300)
    embedded_q = sequence_embed([story['query']])[0] # (qlen, 300)
    enc_query = bow_encode([embedded_q])[0] # (300,)
    # ---------------------------
    # Iterative theorem proving
    rules = list()
    unk = self.xp.zeros(len(word2idx), dtype=self.xp.float32) # Unknown word
    unk[0] = 10.0 # High unnormalised score
    for rs in rule_stories:
      r = self.rulegen(rs) # Differentiable rule generated from story
      vs = {vidx:unk for vidx in r['vmap'].keys()} # Init unknown for every variable
      rules.append((r, vs))
    # Compute iterative updates on variables
    rscores = list() # final rule scores
    for rule, vs in rules:
      # Encode rule
      enc_q = sequence_embed([rule['story']['query']])[0] # (qlen, 300)
      enc_body = sequence_embed(rule['story']['context']) # [(s1len, 300), ...]
      # ---------------------------
      # Iterative proving
      for _ in range(1):
        # Setup variable grounding based on variable state
        rwords = [rule['story']['query']]+rule['story']['context'] # [(qlen,), (s1len,), ...]
        # Gather variable values
        vvalues = [F.vstack([F.softmax(vs[widx], 0) @ wordvecs for widx in widxs]) for widxs in rwords]
                  # [(qlen, 300), (s1len, 300), ...]
        # Merge ground with variable values
        grounds = (enc_q,)+enc_body # [(qlen, 300), (s1len, 300), ...]
        vgates = [F.expand_dims(F.hstack([rule['vmap'][widx] for widx in widxs]), 1)
                  for widxs in rwords] # [(qlen, 1), (s1len, 1), ...]
        qtoprove, *bodytoprove = [vg*vv+(1-vg)*gr for vg, vv, gr in zip(vgates, vvalues, grounds)]
        # ---------------------------
        # Unifications give new variable values and a score for match
        # Unify query
        qscore, qunified = self.unify(qtoprove, [story['query']], (embedded_q,)) # (), (qlen, len(word2idx))
        scores, unifications = [qscore], [qunified]
        # Unify body conditions
        for btoprove in bodytoprove:
          score, unified = self.unify(btoprove, story['context'], embedded_ctx) # (), (snlen, len(word2idx))
          scores.append(score)
          unifications.append(unified)
        # ---------------------------
        # Update variables after unification
        words = np.concatenate(rwords) # (qlen+s1len+s2len+...,)
        unique_idxs = np.unique(words)
        weights = [F.repeat(scores[i], len(seq)) for i, seq in enumerate(rwords)] # [(qlen,), (s1len,), ...]
        weights = F.sigmoid(F.hstack(weights)) # (qlen+s1len+s2len+...,)
        unifications = F.concat(unifications, 0) # (qlen+s1len+s2len+..., len(word2idx))
        # Weighted sum based on sigmoid score
        normalisations = {widx:0.0 for widx in unique_idxs}
        for pidx, widx in enumerate(words):
          normalisations[widx] += weights[pidx]
        # Reset variable states
        for widx in unique_idxs:
          vs[widx] = self.xp.zeros(len(word2idx), dtype=np.float32)
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
    rule, rscore, vs = rules[0], rscores[0], varstates[0]
    # Get ground value
    eye = self.xp.eye(len(word2idx)) # (len(word2idx), len(word2idx))
    asground = eye[rule['story']['answers']] # (len(answers), len(word2idx))
    # Get variable values
    asvar = F.vstack([vs[widx] for widx in rule['story']['answers']]) # (len(answers), len(word2idx))
    # Compute final value using variable gating values
    asvgates = F.hstack([rule['vmap'][widx] for widx in rule['story']['answers']]) # (len(answers),)
    asvgates = F.expand_dims(asvgates, 1) # (len(answers), 1)
    prediction = asvar*asvgates + (1-asvgates)*asground # (len(answers), len(word2idx))
    # Compute final final value using rule score
    noans = self.xp.zeros(prediction.shape) # (len(answers), len(word2idx))
    noans[0,0] = 10.0 # high unnormalised score for unknown word
    answer = rscore*prediction + (1-rscore)*noans # (len(answers), len(word2idx))
    return answer, rscore

# ---------------------------

# Setup model
model = Infer()
optimiser = C.optimizers.Adam().setup(model)

# Stories to generate rules from
rule_repo = [enc_stories[0]]
answers = set()
training = list()
init_tsize = 4
for es in enc_stories[1:]:
  if es['answers'][0] not in answers:
    training.append(es)
    answers.add(es['answers'][0])
  if len(training) == init_tsize:
    break

for sidx, curr_story in enumerate(enc_stories[init_tsize:]):
  answer, confidence = model(curr_story, rule_repo)
  # answer.shape == (len(answers), len(word2idx))
  # Check if correct answer
  prediction = np.argmax(F.softmax(answer).array, axis=-1)[0]
  if prediction == curr_story['answers'][0]:
    continue
  print("# ---------------------------")
  print(stories[sidx+init_tsize])
  print("WRONG:", idx2word[prediction], idx2word[curr_story['answers'][0]])
  # Add to training set
  training.append(curr_story)
  print("TRAINING SIZE:", len(training))
  print("# ---------------------------")
  # ---------------------------
  try:
    # and retrain network
    for epoch in range(1000):
      stime = time.time()
      model.cleargrads()
      outs = [model(ts, rule_repo)[0] for ts in training] # [(1, len(word2idx), ...]
      preds = F.vstack(outs) # (len(training), len(word2idx))
      targets = np.array([ts['answers'][0] for ts in training]) # (len(training),)
      loss = F.softmax_cross_entropy(preds, targets)
      if loss.array < 0.01:
        break
      loss.backward()
      optimiser.update()
      etime = time.time()
      print(f"Epoch: {epoch} Loss: {str(loss.array)} Time: {round(etime-stime,3)}")
  except KeyboardInterrupt:
    import ipdb; ipdb.set_trace()

print("\nTraining complete:")
print("RULES:", rule_repo)
