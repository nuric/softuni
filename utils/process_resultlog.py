"""Process the output of log json files to CSV format."""
import argparse
import json
import os


parser = argparse.ArgumentParser(description="Convert JSON log to CSV.")
parser.add_argument("fpath", nargs='+', help="Log file path.")
parser.add_argument("-hd", "--header", action="store_true", help="Just print the header.")
ARGS = parser.parse_args()

pkeys = ['dataset', 'taskid', 'nrules', 'strong', 'tsize', 'arity', 'runc']
# pylint: disable=line-too-long
keys = ['main/loss', 'main/vmap', 'main/uatt', 'main/oatt', 'main/batt', 'main/rpred', 'main/opred', 'main/uni', 'main/oacc', 'main/acc', 'val/main/loss', 'val/main/vmap', 'val/main/uatt', 'val/main/oatt', 'val/main/batt', 'val/main/rpred', 'val/main/opred', 'val/main/uni', 'val/main/oacc', 'val/main/acc', 'test/main/loss', 'test/main/vmap', 'test/main/uatt', 'test/main/oatt', 'test/main/batt', 'test/main/rpred', 'test/main/opred', 'test/main/uni', 'test/main/oacc', 'test/main/acc', 'epoch', 'iteration', 'elapsed_time']

# Check to print header
if ARGS.header:
  print(*(pkeys + keys), sep=',')
  exit()

for file_path in ARGS.fpath:
  # Parse the filename first
  fname = os.path.splitext(os.path.basename(file_path))[0]
  # fname = fname.split('.')[0] # chop .json
  params = fname.split('_')[:-1] # chop _log
  dataset = params[0][:2] # qa, dl
  dataset = 'babi' if fname[:2] == 'qa' else 'deeplogic'
  taskid = int(params[0][2:])

  # Parse filename
  if dataset == 'babi':
    arity = -1
    nrules = int(params[1][1:])
    strong = int(params[2][1:])
    tsize = int(params[3][1:])
    tsize = 1000 if tsize == 0 else 50
    runc = int(params[4][1:])
  else:
    arity = int(params[1][2:])
    shift = 0
    if params[2] == "10k":
      tsize = 10000
      shift = 1
    else:
      tsize = int(params[4][1:])
      tsize = 1000 if tsize == 0 else 50
    nrules = int(params[2+shift][1:])
    strong = int(params[3+shift][1:])
    runc = int(params[5+shift][1:])

  # Parse json content
  with open(file_path, 'r') as f:
    log = json.load(f)

  # Print json log
  pvals = [dataset, taskid, nrules, strong, tsize, arity, runc]
  for l in log:
    try:
      vals = [l[k] for k in keys]
      print(*(pvals+vals), sep=',')
    except KeyError:
      pass
