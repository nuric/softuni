"""Aggregate results from the csv file."""
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fpath", help="CSV file path.")
ARGS = parser.parse_args()

df = pd.read_csv(ARGS.fpath)

# Sort first by accuracy then by loss
sortdf = df.sort_values(by=['val/main/acc', 'val/main/loss'], ascending=[False, True])
# Take the best epoch
grouped = sortdf.groupby(['dataset', 'taskid', 'nrules', 'strong', 'tsize', 'arity', 'runc']).first()

# Take the best run
df = grouped.reset_index()
sortdf = df.sort_values(by=['val/main/acc', 'val/main/loss'], ascending=[False, True])
grouped = sortdf.groupby(['dataset', 'taskid', 'nrules', 'strong', 'tsize', 'arity']).first()

grouped.to_csv('agg_results.csv')
