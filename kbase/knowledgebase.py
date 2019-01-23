"""KnowledgeBase of rules for NLLog"""
import logging
from operator import itemgetter

log = logging.getLogger(__name__)


class KnowledgeBase:
  """Represents a collection of Rules."""
  MAX_DEPTH = 7

  def __init__(self, rules=None):
    self.rules = rules or list()

  def add_rule(self, rule):
    """Append a rule to the knowledge base."""
    self.rules.insert(0, rule)

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

  def prove(self, exprs, pos=0, depth=0):
    """Prove given expression on the knowledge base."""
    # exprs = [expr1, expr2, ...]
    # Base case
    if not exprs:
      yield 1.0, (None,) # vacuously true
      return
    if depth >= self.MAX_DEPTH:
      log.debug("MAX_DEPTH: %s", ' && '.join(map(repr, exprs)))
      yield 0.0, (None,) # max depth
      return
    # Try to prove
    log.debug("PROVE: %s", ' && '.join(map(repr, exprs)))
    # Save state for backtracking
    state = exprs[0].save_state()
    # Find matching rules
    for sim, rule in self.match(exprs[0], pos=pos):
      # Load state
      exprs[0].load_state(state)
      log.debug("PMATCH: %f -- %s -- %s ", sim, repr(exprs[0]), repr(rule))
      # Unify
      uni_sim, goal_rule = rule.apply(exprs[0], pos)
      # Adjust confidence score
      confidence = exprs[0].proof(sim*uni_sim)
      # Prove from left to right
      for child_conf, child_rules in self.prove(goal_rule.body + exprs[1:], depth=depth+1):
        yield confidence*child_conf, (goal_rule,)+child_rules
    # We exhausted all options
