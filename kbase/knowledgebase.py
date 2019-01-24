"""KnowledgeBase of rules for NLLog"""
import logging
from operator import itemgetter
import numpy as np
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


class KnowledgeBase:
  """Represents a collection of Rules."""
  MAX_DEPTH = 7
  CLUSTER = DBSCAN(eps=0.05, min_samples=1)

  def __init__(self, rules=None):
    self.rules = rules or list()

  def add_rule(self, rule):
    """Append a rule to the knowledge base."""
    self.rules.insert(0, rule)

  def match(self, expr, pos=0):
    """Try to find a close matching rule with expr position pos."""
    sri = []
    # Check across all the rules in the knowledge base
    for i, rule in enumerate(self.rules):
      if (pos >= len(rule) or not rule.exprs[pos] or
          (pos == 0 and rule.query)):
        continue
      s = expr.match(rule.exprs[pos])
      sri.append((rule, s, i))
    if not sri:
      return list() # We have no matches
    # Cluster based on similarity
    log.debug("MATCHES: %s -- %s", repr(expr), sri)
    sims = np.array([r[1] for r in sri]).reshape(-1, 1)
    labels = self.CLUSTER.fit_predict(sims)
    clusters = dict()
    for tup, l in zip(sri, labels):
      clusters.setdefault(l, list()).append(tup)
    # Find most similar cluster
    label, _ = max([(l, max(map(itemgetter(1), c))) for l, c in clusters.items()],
                   key=itemgetter(1))
    matches = clusters[label]
    # Sort rules by position
    matches.sort(key=itemgetter(2))
    return matches

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
    for rule, sim, _ in self.match(exprs[0], pos=pos):
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

  def __repr__(self):
    return '\n'.join(map(repr, self.rules))

  def __str__(self):
    return '\n'.join(map(str, self.rules))
