"""Expressions for KnowledgeBase in KevinAI"""
import logging
from .sent import Sent

log = logging.getLogger(__name__)


class Expr:
  """Parent class of expressions in rules."""
  def __bool__(self):
    # Is it a valid expresssion?
    return False

  @property
  def variables(self):
    """Return variables in expression."""
    return list()

  def match(self, other):
    """Rule selection matching."""
    raise NotImplementedError("Base Expr match.")

  def unify(self, other):
    """Unify variables between expressions."""
    raise NotImplementedError("Base Expr match.")

  def proof(self, confidence):
    """Adjust confidence value based on semantics."""
    # pylint: disable=no-self-use
    return confidence


class ExprNBF(Expr):
  """Negation by failure expression."""
  def __init__(self, expr):
    self.expr = expr or None

  def __bool__(self):
    return bool(self.expr)

  def match(self, other):
    """Match on the inner expression."""
    return self.expr.match(other)

  def unify(self, other):
    """Unify on the inner expression."""
    return self.expr.unify(other)

  def proof(self, confidence):
    """1 minus the proof for negation by failure semantics."""
    return 1 - confidence


class ExprSent(Expr):
  """Holds a parse tree as expression."""
  def __init__(self, sent=None):
    if isinstance(sent, str):
      self.sent = Sent.from_text(sent)
    elif isinstance(sent, Sent):
      self.sent = sent
    else:
      self.sent = Sent.from_text("")

  def __len__(self):
    return len(self.sent)

  def __bool__(self):
    return len(self) > 0

  def __repr__(self):
    return repr(self.sent)

  def __str__(self):
    return str(self.sent)

  @property
  def variables(self):
    """Return the vars from wrapped parse tree."""
    return self.sent.variables

  def match(self, other):
    """Match parse tree based on similarity."""
    # Check if it is a wrapping expression
    if not isinstance(other, type(self)):
      return other.match(self)
    # Sentence similarity
    return self.sent.similarity(other.sent)

  def unify(self, other):
    """Unify parse tree expressions."""
    log.debug("UNIFY: %s -- %s", repr(self), repr(other))
    # Check if it is a wrapping expression
    if not isinstance(other, type(self)):
      return other.unify(self)
    # Unify parse tree
    return self.sent.unify(other.sent)
