"""Utils for KnowledgeBase."""
import numpy as np

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
