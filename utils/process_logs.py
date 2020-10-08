"""Process the output of log json files to CSV format."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert JSON log to CSV.")
parser.add_argument("fpath", nargs='+', help="Log file path.")
ARGS = parser.parse_args()

# List of header keys for the CSV
header_keys = []

for file_path in ARGS.fpath:
  # Parse the filename first
  assert file_path.endswith('_log.json'), "Unknown log file name."
  log_path = Path(file_path)
  runid = log_path.name.split('_')[0] # chop _log

  # Parse run parameters
  with (log_path.parent / (runid + '_params.json')).open() as f:
    params = json.load(f)

  # Parse json content
  with log_path.open() as f:
    log = json.load(f)

  if not header_keys:
    log_keys = list(log[0].keys())
    params_keys = list(params.keys())
    header_keys = sorted(log_keys + params_keys)
    print(*header_keys, sep=',')

  for l in log:
    run_data = dict()
    run_data.update(params)
    run_data.update(l)
    vals = [run_data[k] for k in header_keys]
    print(*vals, sep=',')
