"""Rules in NLLOG for KnowledgeBase"""


class Rule(object):
  """Represents a single applicable rule."""
  def __init__(self, exprs=None):
    self.exprs = exprs or list()

  @property
  def head(self):
    """Return the head of the rule if any."""
    if self.exprs:
      return self.exprs[0]

  @property
  def query(self):
    """Return the query expression if any."""
    if len(self.exprs) >= 2:
      return self.exprs[1]

  @property
  def body(self):
    """Return the non-query body of the rule if any."""
    return self.exprs[2:]

  @property
  def variables(self):
    """Recursively fetch variables inside expression."""
    return [v for e in self.exprs for v in e.variables]

  def unify(self, pos, expr):
    """Unify expression at given position with expr."""
    sim = self.exprs[pos].unify(expr)
    # Update variables
    for v in self.exprs[pos].variables:
      for vv in self.variables:
        if v.name == vv.name:
          vv.value = v.value
    return sim

  def __len__(self):
    return len(self.exprs)

  def __contains__(self, expr):
    return any([expr in child for child in self.exprs])

  def __repr__(self):
    if self.exprs:
      return repr(self.exprs[0]) + ' <- ' + ' | '.join(map(repr, self.exprs[1:]))
    return ' <BLANK- '

  __str__ = __repr__
