"""bAbI run of NLLog"""
import argparse
import logging
from kbase.expr import ExprSent
from kbase.rule import Rule
from kbase.knowledgebase import KnowledgeBase


# Arguments
parser = argparse.ArgumentParser(description="Run NLLog on bAbI tasks.")
parser.add_argument("task", help="File that contains task.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output.")
ARGS = parser.parse_args()

# Debug
if ARGS.debug:
  logging.basicConfig(level=logging.DEBUG)

stories = [[]]
# Load in task
with open(ARGS.task) as f:
  prev_id = 0
  for line in f:
    line = line.strip()
    sid, sl = line.split(' ', 1)
    # Is this a new story?
    sid = int(sid)-1
    if sid < prev_id:
      stories.append(list())
    # Check for question or not
    if '\t' in sl:
      q, a, supps = sl.split('\t')
      stories[-1].append((q.strip(), a,
                          [int(sup)-1 for sup in supps.split()]))
    else:
      # Just a statement
      stories[-1].append(sl)
    prev_id = sid
    if len(stories) > 2:
      break
print("SAMPLE:", stories[0])

p1 = ExprSent("X:Bob went to the Y:hallway.")
q0 = ExprSent("Where is X:Bob?")
a0 = ExprSent("Y:hallway")
r = Rule([a0, q0, p1])

for story in stories:
  kb = KnowledgeBase([r])
  for line in story:
    # Check statement or question
    if isinstance(line, str):
      # Statement
      kb.add_rule(Rule([ExprSent(line)]))
      continue
    # We have a question
    q, a, sups = line
    confidence, rules = next(kb.prove([ExprSent(q)], 1))
    prediction = str(rules[0].head)
    if prediction != a:
      # We got a wrong answer
      print("-----")
      print(kb)
      print("-----")
      print(rules)
      print("-----")
      print("ANSWER:", rules[0].head, confidence)
      print("EXPECTED:", a, sups)
      print("-----")
      exit()
print("--ALL PASS--")
