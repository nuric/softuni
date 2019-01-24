"""Representation of a single Sentence."""
import logging
import re
from operator import itemgetter
import numpy as np
from .token import WordToken, VarToken
from .utils import tokenise, cosine_similarity

log = logging.getLogger(__name__)


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
          tokens[-1].default.text += ' ' + word
        else:
          # We have a new variable token
          tokens.append(VarToken(varname, default=WordToken(word)))
      else:
        # We have a just a word token
        tokens.append(WordToken(token))
    return cls(tokens)

  @property
  def variables(self):
    """Return tuple of variables in order."""
    return self.tokens, [i for i, v in enumerate(self.tokens) if isinstance(v, VarToken)]

  def copy(self):
    """Return re-usable sentence object."""
    return Sent([t.copy() for t in self.tokens])

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
      weights.append(4.0 if isinstance(t, VarToken) and t.value else 1.0)
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

  def __contains__(self, token):
    return token in self.tokens

  def similarity(self, other):
    """Calculate similarity to other sentence."""
    if not self or not other:
      return 0.0
    return cosine_similarity(self.vector, other.vector)

  def clear_variables(self):
    """Clear all variable bindings."""
    vl, vidxs = self.variables
    for i in vidxs:
      vl[i].value = None

  def unify(self, other):
    """Bind the variables of this sent with possible matches of other."""
    if not self.variables[1] or not other:
      return 1.0
    # A naive semantic unification
    sims = list()
    vl, vidxs = self.variables
    for i in vidxs:
      if vl[i] in other or vl[i].value:
        continue
      # find maximal match in other
      simtokens = sorted([(vl[i].similarity(t), t) for t in other], key=itemgetter(0), reverse=True)
      for sim, token in simtokens:
        # Ensure it is not a token we contain and that is already bound to a variable
        if (token in self or
            any([vl[j].similarity(token) > 0.95 for j in vidxs if vl[j].value])):
          continue # find another token
        sims.append(sim)
        log.debug("BIND: %s << %s, %f", repr(vl[i]), repr(token), sim)
        if isinstance(token, VarToken):
          token.name = vl[i].name # preserve name for normalisation
          token.default = vl[i].default # preserve default for similarity
          vl[i] = token # replace with that variables
        else:
          vl[i].value = token # just bind token value
        break
    return np.mean(sims) if sims else 1.0
