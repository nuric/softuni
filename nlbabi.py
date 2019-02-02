"""bAbI run on neurolog."""
import argparse
import logging
import numpy as np
import chainer as C
import chainer.links as L
import chainer.functions as F


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
      stories.append({'context': context.copy(),
                      'query': q, 'answers': a.split(',')})
    else:
      # Just a statement
      context.append(sl)
    prev_id = sid
print("TOTAL:", len(stories), "stories")
print("SAMPLE:", stories[0])

# ---------------------------

# Load word vectors
wordvecs = list()
word2idx = dict()
with open(ARGS.vocab) as f:
  for i, l in enumerate(f):
    word, *vec = l.split(' ')
    wordvecs.append(np.array([float(n) for n in vec]))
    word2idx[word] = i
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
    assert len(allwords) == len(inverse_idxs), "Convolved features do not match story len."
    # Add whether they appear more than once
    appeartwice = (unique_counts[inverse_idxs] > 1) # (qlen+s1len+s2len+...,)
    appeartwice = appeartwice.astype(np.float32).reshape(-1, 1) # (qlen+s1len+s2len+..., 1)
    allwords = F.concat([allwords, appeartwice], axis=1) # (qlen+s1len+s2len+..., 17)
    wordvars = self.isvariable_linear(allwords) # (qlen+s1len+s2len+..., 1)
    wordvars = F.squeeze(wordvars, 1) # (qlen+s1len+s2len+...,)
    wordvars = F.sigmoid(wordvars) # (qlen+s1len+s2len+...,)
    # Merge word variable predictions
    iswordvar = {idx:1.0 for idx in unique_idxs}
    for idx in inverse_idxs:
      iswordvar[unique_idxs[idx]] *= wordvars[idx]
    # ---------------------------
    # Tells whether a context is in the body, negated or not
    # and whether a word is a variable or not
    return {'story': story, 'bodymap': ctxinbody, 'vmap': iswordvar}
rulegen = RuleGen()

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

  def forward(self, toprove, candidates):
    """Given two sentences compute variable matches and score."""
    # toprove.shape = (plen, 300), candidates = [(s1len, 300), (s2len, 300), ...]
    # ---------------------------
    # Calculate a match for every word in s1 to every word in s2
    # Compute contextual representations
    cwords = [F.squeeze(self.convolve_words(F.expand_dims(s.T, 0)), 0).T
              for s in (toprove,)+candidates] # [(plen,16), (s1len,16), (s2len,16]
    # Compute similarity between every candidate
    ctp, *ccandids = cwords # (plen,16), [(s1len,16), (s2len,16), ...]
    sims = [F.softmax(ctp @ c.T, axis=1) for c in ccandids] # [(plen,s1len), (plen,s2len), ...]
    # ---------------------------
    # Calculate score for candidate matches
    # Compute bag of words
    pbow, *cbows = [F.expand_dims(F.sum(s, axis=0), 0) for s in cwords] # [(1,16), (1,16)]
    pbow = F.expand_dims(pbow, 0) # (1,1,16) (layers, batchsize, 16)
    cbows = F.concat(cbows, axis=0) # (len(candidates), 16)
    _, raw_scores = self.match_rnn(pbow, [cbows]) # _, [(len(candidates), 16)]
    raw_scores = self.match_linear(raw_scores[0]) # (len(candidates), 1)
    raw_scores = F.squeeze(raw_scores, 1) # (len(candidates),)
    # ---------------------------
    # Calculate attended unififed word representations for toprove
    uwords = F.concat([F.expand_dims(s @ c, 0) for s, c in zip(sims, candidates)], 0) # (len(candidates), plen, 300)
    # Weighted sum using scores
    weights = F.softmax(raw_scores, 0) # (len(candidates),)
    final_words = F.einsum("i,ijk->jk", weights, uwords) # (plen, 300)
    final_score = F.max(raw_scores) # ()
    return final_score, final_words

# ---------------------------

# Inference network
class Infer(C.Chain):
  """Takes a story, a set of rules and predicts answers."""
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.unify = Unify()

  def forward(self, story, rules):
    """Given story and rules predict answers."""
    # Encode story
    embedded_ctx = sequence_embed(story['context']) # [(s1len, 300), (s2len, 300), ...]
    enc_ctx = bow_encode(embedded_ctx) # [(300,), (300,), ...]
    enc_ctx = F.vstack(enc_ctx) # (clen, 300)
    embedded_q = sequence_embed([story['query']])[0] # (qlen, 300)
    enc_query = bow_encode([embedded_q])[0] # (300,)
    # ---------------------------
    # Iterative theorem proving
    # Initialise variable states
    varstates = [{vidx:self.xp.zeros(300, dtype=np.float32) for vidx in r['vmap'].keys()}
                for r in rules]
    # Compute iterative updates on variables
    rscores = list() # final rule scores
    for ridx, r in enumerate(rules):
      # Encode rule
      enc_q = sequence_embed([r['story']['query']])[0] # (qlen, 300)
      enc_body = sequence_embed(r['story']['context']) # [(s1len, 300), ...]
      # ---------------------------
      # Setup variable grounding based on variable state
      vs = varstates[ridx]
      # Iterative proof
      for _ in range(1):
        rwords = [r['story']['query']]+r['story']['context'] # [(qlen,), (s1len,), ...]
        # Gather variable values
        vvalues = [F.vstack([vs[widx] for widx in widxs]) for widxs in rwords]
                  # [(qlen, 300), (s1len, 300), ...]
        # Merge ground with variable values
        grounds = (enc_q,)+enc_body # [(qlen, 300), (s1len, 300), ...]
        vgates = [F.expand_dims(F.hstack([r['vmap'][widx] for widx in widxs]), 1)
                  for widxs in rwords]
                  # [(qlen, 1), (s1len, 1), ...]
        qtoprove, *bodytoprove = [vg*vv+(1-vg)*gr for vg, vv, gr in zip(vgates, vvalues, grounds)]
        # ---------------------------
        # Unifications give new variable values and a score for match
        # Unify query
        qscore, qunified = self.unify(qtoprove, (embedded_q,)) # (), (qlen, 300)
        scores, unifications = [qscore], [qunified]
        # Unify body conditions
        for btoprove in bodytoprove:
          score, unified = self.unify(btoprove, embedded_ctx) # (), (snlen, 300)
          scores.append(score)
          unifications.append(unified)
        # ---------------------------
        # Update variables after unification
        words = np.concatenate(rwords) # (qlen+s1len+s2len+...,)
        unique_idxs, unique_counts = np.unique(words, return_counts=True)
        weights = [F.repeat(scores[i], len(seq)) for i, seq in enumerate(rwords)] # [(qlen,), (s1len,), ...]
        weights = F.sigmoid(F.hstack(weights)) # (qlen+s1len+s2len+...,)
        unifications = F.concat(unifications, 0) # (qlen+s1len+s2len+..., 300)
        # Weighted sum based on sigmoid score
        normalisations = {widx:0.0 for widx in unique_idxs}
        for pidx, widx in enumerate(words):
          normalisations[widx] += weights[pidx]
        for pidx, widx in enumerate(words):
          varstates[ridx][widx] += (unifications[pidx] / normalisations[widx])
      # ---------------------------
      # End iterations with final score for each premise
      prem_scores = F.sigmoid(F.hstack(scores)) # (1 + len(body),)
    # ---------------------------
    # Compute overall score for rule
    # r['bodymap'].shape == (len(body), 2) => inbody, isnegated
    qscore, bscores = prem_scores[0], prem_scores[1:]
    import ipdb; ipdb.set_trace()
    print("HERE")
    # ---------------------------
    return "infer"
infer = Infer()

# ---------------------------

# Stories to generate rules from
repo = [enc_stories[0]]
# Train loop
for dstory in enc_stories[1:]:
  # Get rules based on seen stories
  rules = [rulegen(s) for s in repo]
  answer = infer(dstory, rules)
  print("RULES:", rules)
  break
