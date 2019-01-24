"""bAbI run of NLLog"""
import argparse
import logging
import string
from kbase.expr import ExprSent
from kbase.sent import STOPWORDS
from kbase.rule import Rule
from kbase.knowledgebase import KnowledgeBase
from kbase.utils import tokenise


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
print("TOTAL:", len(stories), "stories")
print("SAMPLE:", stories[0])

def one_shot(query, answer, sups, story):
  """Create a one shot rule."""
  q, a, sups = tokenise(query), tokenise(answer), [tokenise(story[i]) for i in sups]
  vnames = list(string.ascii_lowercase)
  vmap = dict()
  # Convert common tokens into variables
  sents = [a, q]+sups
  for i, sent in enumerate(sents):
    rtokens = [t for s in sents[i+1:] for t in s]
    for token in sent:
      if token in STOPWORDS:
        continue
      if token in rtokens:
        vmap.setdefault(token, vnames.pop())
  # Construct rule
  sents = [' '.join([vmap[t]+':'+t if t in vmap else t for t in s])
           for s in sents]
  return Rule([ExprSent(s) for s in sents])

# --------------------
# p1 = ExprSent("X:Bob went to the Y:hallway.")
# q0 = ExprSent("Where is X:Bob?")
# a0 = ExprSent("Y:hallway")
# r = Rule([a0, q0, p1])
learned_rules = list()

for story in stories:
  kb = KnowledgeBase(learned_rules.copy())
  for line in story:
    # Check statement or question
    if isinstance(line, str):
      # Statement
      kb.add_rule(Rule([ExprSent(line)]))
      continue
    # We have a question
    q, a, sups = line
    # Obtain prediction
    confidence, prediction = 0.0, None
    try:
      confidence, rules = next(kb.prove([ExprSent(q)], 1))
      prediction = str(rules[0].head)
    except StopIteration as e:
      print(e) # No more solutions
    # Check and learn
    if prediction != a:
      # We got a wrong answer
      print("----KNOWLEDGEBASE----")
      print(kb)
      print("-----")
      print("QUERY:", q)
      print("ANSWER:", prediction, confidence)
      print("EXPECTED:", a, sups)
      print("-----")
      # Attempt to one-shot learn
      rule = one_shot(q, a, sups, story)
      print("NEWRULE:", rule)
      learned_rules.append(rule)
      kb.rules.append(rule)
print("--ALLPASS--")
print("-----------")
print("--LEARNED--")
for r in learned_rules:
  print(r)
