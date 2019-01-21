"""Representation of a single Sentence."""
import re
from operator import itemgetter
import numpy as np
from .token import WordToken, VarToken
from .utils import tokenise, cosine_similarity


class Sent:
  """Single Sentence composed of tokens and variables."""
  VAR_RE = r'[a-z]+:'

  def __init__(self, tokens=None):
    self.tokens = tokens or list()

  @classmethod
  def from_text(cls, text):
    """Parse given text into tokens with variables."""
    raw_tokens = tokenise(text.strip())
    tokens = list()
    for token in raw_tokens:
      # Check for annotated variable
      if re.match(cls.VAR_RE, token):
        varname, word = token.split(':')
        # Check for merger
        if tokens and isinstance(tokens[-1], VarToken) \
           and tokens[-1].name == varname:
          tokens[-1].ground.text += ' ' + word
        else:
          # We have a new variable token
          tokens.append(VarToken(varname, ground=WordToken(word)))
      else:
        # We have a just a word token
        tokens.append(WordToken(token))
    return cls(tokens)

  @property
  def variables(self):
    """Return tuple of variables in order."""
    return [t for t in self.tokens if isinstance(t, VarToken)]

  @property
  def vector(self):
    """Return sentence vector."""
    # weighted bag of words calculation, variables with values take precedence
    weights, vectors = list(), list()
    for t in self:
      if t.vector is None or not any(t.vector):
        continue
      vectors.append(t.vector)
      # Weight towards bound variables
      weights.append(2.1 if isinstance(t, VarToken) and t.value else 1.0)
    # Softmax
    weights = np.exp(weights)
    weights /= np.sum(weights)
    return np.average(vectors, axis=0, weights=weights)

  def __getitem__(self, idx):
    return self.tokens[idx]

  def __iter__(self):
    return iter(self.tokens)

  def __len__(self):
    return len(self.tokens)

  def __repr__(self):
    return ' '.join(map(repr, self.tokens))

  def __str__(self):
    return ' '.join(map(str, self.tokens))

  def similarity(self, other):
    """Calculate similarity to other sentence."""
    if not self or not other:
      return 0.0
    return cosine_similarity(self.vector, other.vector)

  def clear_variables(self):
    """Clear all variable bindings."""
    for v in self.variables:
      v.value = None

  def unify(self, other):
    """Bind the variables of this sent with possible matches of other."""
    if not self.variables or not other:
      return 0.0
    # A naive semantic unification
    sims = list()
    for v in self.variables:
      # find maximal match in other
      sim, token = max([(v.similarity(t), t) for t in other], key=itemgetter(0))
      sims.append(sim)
      v.value = token
    return sum(sims)/len(sims)
