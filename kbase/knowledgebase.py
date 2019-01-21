"""KnowledgeBase of rules for NLLog"""
import logging
from operator import itemgetter

log = logging.getLogger(__name__)


class KnowledgeBase:
  """Represents a collection of Rules."""
  def __init__(self, rules=None):
    self.rules = rules or list()

  def add_rule(self, rule):
    """Append a rule to the knowledge base."""
    self.rules.insert(0, rule)

  @property
  def variables(self):
    """Return all variables present in knowledge base."""
    return [v for r in self.rules for v in r.variables]

  def save_vars(self):
    """Return a dictionary of variables to value references."""
    return {v:v.value for v in self.variables}

  def load_vars(self, state):
    """Assign variable values from dictionary references."""
    for v in self.variables:
      v.value = state.get(v, v.value)

  def reset(self):
    """Reset the state of the knowledge base."""
    # Clear variable bindings
    for v in self.variables:
      v.value = None

  def match(self, expr, pos=0):
    """Try to find a close matching rule with expr position pos."""
    sr = []
    # Check across all the rules in the knowledge base
    for rule in self.rules:
      if pos >= len(rule) or not rule.exprs[pos]:
        continue
      s = expr.match(rule.exprs[pos])
      sr.append((s, rule))
    sr.sort(key=itemgetter(0), reverse=True) # most similar is first
    return sr

  def prove(self, exprs, pos=0):
    """Prove given expression on the knowledge base."""
    # Base case
    if not exprs:
      yield 1.0, (None,) # vacuously true
      return
    # Try to prove
    log.debug("PROVE: %s", '<>'.join(map(repr, exprs)))
    # Save state for backtracking
    state = self.save_vars()
    # Find matching rules
    for sim, rule in self.match(exprs[0], pos=pos):
      log.debug("PMATCH: %f -- %s -- %s ", sim, repr(exprs[0]), repr(rule))
      self.load_vars(state)
      # Unify
      confidence = exprs[0].proof(sim*rule.unify(pos, exprs[0]))
      # Prove from left to right
      for child_conf, child_rules in self.prove(rule.body + exprs[1:]):
        yield confidence*child_conf, (rule,)+child_rules
    # We exhausted all options
