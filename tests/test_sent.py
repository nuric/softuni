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
