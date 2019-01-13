"""Utility test cases."""
import unittest
import numpy as np
import kbase.utils as utils


class TestUtils(unittest.TestCase):
  """Test cases for utility functions."""
  def test_cosine_similarity_equal(self):
    """Check the cosine similarity of equal vectors."""
    x = np.arange(4)
    self.assertEqual(utils.cosine_similarity(x,x), 1.0)

  def test_cosine_similarty_sim(self):
    """Check the cosine similarity of similar vectors."""
    x, y = np.arange(4), np.array([1,2,3,4])
    self.assertAlmostEqual(utils.cosine_similarity(x,y), 0.9759000729485332)

  def test_cosine_similarty_zero(self):
    """Check the cosine similarity of zero vector."""
    x, y = np.zeros(4), np.array([1,2,3,4])
    self.assertEqual(utils.cosine_similarity(x,y), 0.0)

  def test_tokenisation_lower(self):
    """Check the tokens of the tokenise sentence."""
    ts = utils.tokenise("X:Alice went to the kitchen.")
    tokens = ['x:alice', 'went', 'to', 'the', 'kitchen']
    self.assertSequenceEqual(ts, tokens)

  def test_tokenisation_upper(self):
    """Check the tokens of the tokenise sentence."""
    ts = utils.tokenise("Alice went to the Kitchen.", lower=False)
    tokens = ['Alice', 'went', 'to', 'the', 'Kitchen']
    self.assertSequenceEqual(ts, tokens)

  def test_tokenisation_empty(self):
    """Check the tokens of the tokenise sentence."""
    self.assertSequenceEqual(utils.tokenise(""), [])

  def test_empty_word_vector(self):
    """Check empty word vector functions."""
    w = utils.WordVectors()
    self.assertEqual(len(w), 0)
    self.assertEqual(w.dim, 0)

  def test_word_vector_parsing(self):
    """Check if the word vector stream parses correctly."""
    w = utils.WordVectors(["the 0.5 0.6 0.1", "and 0.4 0.3 0.7"])
    self.assertEqual(len(w), 2)
    self.assertEqual(w.dim, 3)
    self.assertTrue((w['the'] == np.array([0.5, 0.6, 0.1])).all())
    self.assertTrue((w['and'] == np.array([0.4, 0.3, 0.7])).all())
    self.assertTrue((w['42'] == np.zeros(w.dim)).all())
