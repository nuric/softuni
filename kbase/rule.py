"""Rules in NLLOG for KnowledgeBase"""


class Rule:
  """Represents a single applicable rule."""
  def __init__(self, exprs=None):
    self.exprs = exprs or list()
    self.normalise_vars(self.exprs)

  @property
  def head(self):
    """Return the head of the rule if any."""
    return self.exprs[0] if self.exprs else None

  @property
  def query(self):
    """Return the query expression if any."""
    return self.exprs[1] if len(self.exprs) >= 2 else None

  @property
  def body(self):
    """Return the non-query body of the rule if any."""
    return self.exprs[2:]

  @staticmethod
  def normalise_vars(exprs):
    """Re-bind matching variable tokens of expressions."""
    vmap = dict()
    for e in exprs:
      vlist, vidxs = e.variables
      for i in vidxs:
        vlist[i] = vmap.setdefault(vlist[i].name, vlist[i])

  def apply(self, expr, pos):
    """Apply rule at given position to return new rule."""
    # Create re-usable expressions to bind variables etc
    cexprs = [e.copy() for e in self.exprs]
    self.normalise_vars(cexprs)
    # Unify both ways
    uni_sim = cexprs[pos].unify(expr)
    uni_sim *= expr.unify(cexprs[pos])
    # Substitute VarTokens after unification
    return uni_sim, Rule(cexprs)

  def __len__(self):
    return len(self.exprs)

  def __contains__(self, expr):
    return any([expr in child for child in self.exprs])

  def __repr__(self):
    if self.exprs:
      return repr(self.exprs[0]) + ' <- ' + ' | '.join(map(repr, self.exprs[1:]))
    return ' <BLANK- '

  __str__ = __repr__
