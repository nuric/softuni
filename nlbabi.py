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
wordvecs = np.array(wordvecs)
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
      self.linear1 = L.Linear(2, 3)

  def forward(self, story):
    """Given a story generate a probabilistic learnable rule."""
    # Encode sequences
    embedded_ctx = sequence_embed(story['context']) # [(s1len, 300), (s2len, 300), ...]
    enc_ctx = bow_encode(embedded_ctx) # [(300,), (300,), ...]
    enc_ctx = F.concat([F.expand_dims(e, 0) for e in enc_ctx], axis=0) # (clen, 300)
    embedded_q = sequence_embed([story['query']])[0] # (qlen, 300)
    enc_query = bow_encode([embedded_q])[0] # (300,)
    print(enc_query)
    # import pdb; pdb.set_trace()
    # Need:
    # - whether a fact is in the body or not, negated or not, multi-label -> (clen, 2)
    # - whether each word in story is a variable, multi-class
    #   -> {'answer':(alen, 6), 'query':(qlen, 6), ...}
    print("RULEGEN:", self.links(), story)
    return "new rule"
rulegen = RuleGen()

# ---------------------------

# Stories to generate rules from
repo = [enc_stories[0]]
# Train loop
for dstory in enc_stories[1:]:
  # Get rules based on seen stories
  rules = [rulegen(s) for s in repo]
  print("RULES:", rules)
  break
