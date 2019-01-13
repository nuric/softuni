"""Representation of a single Sentence."""
import logging
import re
from .token import WordToken, VarToken
from .utils import tokenise

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
    return tuple(t for t in self.tokens if isinstance(t, VarToken))

  def __len__(self):
    return len(self.tokens)

  def __repr__(self):
    return ' '.join(map(repr, self.tokens))

  def __str__(self):
    return ' '.join(map(str, self.tokens))
