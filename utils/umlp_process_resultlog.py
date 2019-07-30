"""Process the output of log json files to CSV format."""
import argparse
import json
import os

parser = argparse.ArgumentParser(description="Convert JSON log to CSV.")
parser.add_argument("fpath", nargs='+', help="Log file path.")
parser.add_argument("-hd", "--header", action="store_true", help="Just print the header.")
ARGS = parser.parse_args()

pkeys = ['name', 'length', 'symbols', 'invariants', 'embed', 'fold']
keys = [
  "main/uloss",
  "main/uacc",
  "main/igloss",
  "main/igacc",
  "main/oloss",
  "main/oacc",
  "main/vloss",
  "test/main/uloss",
  "test/main/uacc",
  "test/main/igloss",
  "test/main/igacc",
  "test/main/oloss",
  "test/main/oacc",
  "test/main/vloss",
  "epoch",
  "iteration",
  "elapsed_time"
]

# Check to print header
if ARGS.header:
  print(*(pkeys + keys), sep=',')
  exit()

for file_path in ARGS.fpath:
  # Parse the filename first
  fname = os.path.splitext(os.path.basename(file_path))[0]
  params = fname.split('_')[:-1] # chop _log
  num_params = [int(p[1:]) for p in params[1:]]

  # Parse json content
  with open(file_path, 'r') as f:
    log = json.load(f)

  pvals = [params[0]] + num_params
  # for l in log:
  for l in log[-1:]:
    try:
      vals = [l[k] for k in keys]
      print(*(pvals+vals), sep=',')
    except KeyError:
      pass
