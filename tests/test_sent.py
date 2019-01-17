"""Sentence test cases."""
import unittest
from kbase.sent import Sent


class TestSent(unittest.TestCase):
  """Test cases for sentence based operations."""
  def test_creation_from_text_novar(self):
    """Check parsing on no variable sentence."""
    s = Sent.from_text("Alice went to the hallway.")
    self.assertEqual(len(s), 5)
    self.assertFalse(s.variables)

  def test_creation_from_text_var(self):
    """Check parsing with variables."""
    s = Sent.from_text("X:Alice went to the Y:hallway.")
    self.assertEqual(len(s), 5)
    self.assertEqual(len(s.variables), 2)

  def test_creation_from_text_merged(self):
    """Check parsing with merging variables."""
    s = Sent.from_text("How is the weather in X:New X:York?")
    self.assertEqual(len(s), 6)
    self.assertEqual(len(s.variables), 1)

  def test_empty_similarity(self):
    """Similarity is zero if sentence is empty."""
    s = Sent.from_text("")
    o = Sent.from_text("Alice went to the kitchen.")
    self.assertEqual(s.similarity(o), 0.0)

  def test_relative_similarity(self):
    """Similarity is higher for closer sentences."""
    s = Sent.from_text("Bob journeyed to the bathroom.")
    o = Sent.from_text("Alice went to the kitchen.")
    t = Sent.from_text("Alice picked up the ball.")
    self.assertGreater(s.similarity(o), s.similarity(t))

  def test_identitical_similarity(self):
    """Similarity with identitical tokens."""
    s = Sent.from_text("Bob journeyed to the Y:bathroom.")
    self.assertGreater(s.similarity(s), 0.95)

  def test_novar_unification(self):
    """Check if no variable sentence fails to unify."""
    s = Sent.from_text("")
    o = Sent.from_text("Alice went to the kitchen.")
    self.assertEqual(s.unify(o), 0.0)

  def test_empty_unification(self):
    """Check if unification fails with empty string."""
    s = Sent.from_text("X:Alice went to the Y:kitchen.")
    o = Sent.from_text("")
    self.assertEqual(s.unify(o), 0.0)

  def test_single_var_unify(self):
    """Check if unification matches correct token."""
    s = Sent.from_text("Alice went to the Y:kitchen.")
    o = Sent.from_text("Bob journeyed to the bathroom.")
    self.assertGreater(s.unify(o), 0.1)
    self.assertEqual(str(s[-1]), "bathroom")

  def test_var_clear(self):
    """Check if unification matches correct token."""
    s = Sent.from_text("Alice went to the Y:kitchen.")
    o = Sent.from_text("Bob journeyed to the bathroom.")
    self.assertGreater(s.unify(o), 0.1)
    s.clear_variables()
    self.assertEqual(str(s[-1]), "None")

  def test_double_var_unify(self):
    """Check if unification matches 2 correct tokens."""
    s = Sent.from_text("X:Alice went to the Y:kitchen.")
    o = Sent.from_text("Bob journeyed to the bathroom.")
    self.assertGreater(s.unify(o), 0.1)
    self.assertEqual(str(s[0]), "bob")
    self.assertEqual(str(s[-1]), "bathroom")

  def test_nested_unification(self):
    """Check if variables bind to other variables."""
    s = Sent.from_text("X:Alice went to the Y:kitchen.")
    o = Sent.from_text("X:Bob journeyed to the Y:bathroom.")
    r = Sent.from_text("Charlie travelled to the garden.")
    self.assertGreater(s.unify(o), 0.1)
    self.assertGreater(o.unify(r), 0.1)
    self.assertEqual(str(s[0]), "charlie")
    self.assertEqual(str(s[-1]), "garden")
