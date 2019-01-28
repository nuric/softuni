"""bAbI run on neurolog."""
import argparse
import logging
import numpy as np
import chainer as C
import chainer.links as L


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
    print("RULEGEN:", self.links(), story)
    return "new rule"
rulegen = RuleGen()

# ---------------------------

# Stories to generate rules from
repo = [stories[0]]
# Train loop
for dstory in stories[1:]:
  # Get rules based on seen stories
  rules = [rulegen(s) for s in repo]
  print("RULES:", rules)
  break
