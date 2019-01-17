"""Contains Token definition used in Sent."""
from .utils import cosine_similarity, WordVectors


class Token:
  """Base class for tokens in sentences."""
  vector = 0.0

  def similarity(self, other):
    """Calculate similarity to other token."""
    return cosine_similarity(self.vector, other.vector)


class WordToken(Token):
  """Represents a single word token."""
  word2vec = WordVectors.from_file("data/numberbatch.txt")

  def __init__(self, text, vector=None):
    self.text = text
    self.vector = vector or self.word2vec[text]

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

  def __eq__(self, other):
    if isinstance(other, type(self)):
      return self.name == other.name
    return False

  def __hash__(self):
    return hash(self.name)
