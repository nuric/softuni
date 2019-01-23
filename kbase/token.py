"""Contains Token definition used in Sent."""
from .utils import cosine_similarity, WordVectors


class Token:
  """Base class for tokens in sentences."""
  word2vec = WordVectors.from_file("data/wordvecs.txt")
  # numberbatch = WordVectors.from_file("data/numberbatch.txt")
  vector = 0.0

  def copy(self):
    """Return re-usable copy of token."""
    return self # There are no dynamic components

  def similarity(self, other):
    """Calculate similarity to other token."""
    return cosine_similarity(self.vector, other.vector)


class WordToken(Token):
  """Represents a single word token."""
  def __init__(self, text, vector=None):
    self.text = text
    self.vector = vector or self.word2vec[text.lower().replace(' ', '_')]
    # if text not in self.word2vec:
      # self.vector = self.numberbatch[text.lower().replace(' ', '_')]
      # with open("data/wordvecs.txt", 'a') as f:
        # f.write(text + ' ' + ' '.join(map(str, self.vector)) + '\n')

  def __repr__(self):
    return '<' + self.text + '>'

  def __str__(self):
    return self.text


class VarToken(Token):
  """Represents an bindable variable token."""
  def __init__(self, name, value=None, default=None):
    self.name = name
    self.value = value
    self.default = default

  def copy(self):
    """Create a re-usable copy of the variable."""
    return VarToken(self.name, self.value, self.default)

  @property
  def vector(self):
    """Variable token word vector."""
    return self.value.vector if self.value else self.default.vector

  def __repr__(self):
    return '(' + ':'.join((self.name, repr(self.value), repr(self.default))) + ')'

  def __str__(self):
    return str(self.value)
