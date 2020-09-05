#!/bin/bash
# This script wraps to generate programs with varying parameters
FUNC=$1
DDIR='data/deeplogic/'
DCMD='python3 gen_logic.py'
shift
SIZE=$1
#TSIZE=$((SIZE / 10))
TSIZE=1000
shift
ARGS="-pl 1 -cl 1 -ns 2"

all() {
  echo "Generating all tasks..."
  for i in {1..12}; do
    F=$DDIR'train_'${SIZE::-3}'k_task'$i.txt
    TF=$DDIR'test_'${SIZE::-3}'k_task'$i.txt
    echo Writing to $F - $TF
    rm -f $F $TF
    $DCMD $ARGS -t $i -s $SIZE >> $F
    $DCMD $ARGS -t $i -s $TSIZE >> $TF
  done
}

# Run given function
mkdir -p $DDIR
$FUNC "$@"
