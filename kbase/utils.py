"""Utils for KnowledgeBase."""
import logging
import numpy as np

log = logging.getLogger(__name__)

# Utils for similarity measure
def cosine_similarity(vecx, vecy):
  """Calculate cosine similarity between two vectors."""
  d = vecx.dot(vecy)
  norm = np.linalg.norm(vecx) * np.linalg.norm(vecy)
  return d / norm if d else d

# Utils for parsing text
def tokenise(text, filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n',
             lower=True, split=' '):
  """Converts a text to a sequence of words (or tokens).

  # Arguments
    text: Input text (string).
    filters: list (or concatenation) of characters to filter out, such as
        punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
        includes basic punctuation, tabs, and newlines.
    lower: boolean. Whether to convert the input to lowercase.
    split: str. Separator for word splitting.

  # Returns
    A list of words (or tokens).
  """
  if lower:
    text = text.lower()
  translate_dict = dict((c, split) for c in filters)
  translate_map = str.maketrans(translate_dict)
  text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]


class WordVectors():
  """Word vectors dictionary wrapper."""
  def __init__(self, lines=None):
    self.word2vec = dict()
    # word 0.323 0.2323 ...
    for l in (lines or list()):
      word, *vec = l.split(' ')
      self.word2vec[word] = np.array([float(n) for n in vec])
    log.info("Loaded %d word vectors.", len(self.word2vec))
    self.dim = len(next(iter(self.word2vec.values()))) if self.word2vec else 0

  def __getitem__(self, word):
    return self.word2vec.get(word, np.zeros(self.dim))

  def __len__(self):
    return len(self.word2vec)

  @classmethod
  def from_file(cls, fname):
    """Load word vectors from file."""
    with open(fname, 'r') as f:
      return cls(f)
