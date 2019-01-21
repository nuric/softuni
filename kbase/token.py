"""Contains Token definition used in Sent."""
from .utils import cosine_similarity, WordVectors


class Token:
  """Base class for tokens in sentences."""
  word2vec = WordVectors.from_file("data/wordvecs.txt")
  vector = 0.0

  def similarity(self, other):
    """Calculate similarity to other token."""
    return cosine_similarity(self.vector, other.vector)


class WordToken(Token):
  """Represents a single word token."""
  def __init__(self, text, vector=None):
    self.text = text
    self.vector = vector or self.word2vec[text.lower().replace(' ', '_')]

  def __repr__(self):
    return '<' + self.text + '>'

  def __str__(self):
    return self.text


class VarToken(Token):
  """Represents an bindable variable token."""
  def __init__(self, name, value=None, ground=None):
    self.name = name
    self.value = value
    self.ground = ground

  @property
  def vector(self):
    """Variable token word vector."""
    return self.value.vector if self.value else self.ground.vector

  def __repr__(self):
    return ':'.join((self.name, repr(self.value), repr(self.ground)))

  def __str__(self):
    return str(self.value)
