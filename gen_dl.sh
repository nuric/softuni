#!/bin/bash
# This script wraps to generate programs with varying parameters
FUNC=$1
DDIR='data/deeplogic/'
DCMD='python3 data_gen.py'
shift
SIZE=$1
TSIZE=$((SIZE / 10))
shift
ARITY=2
ARGS="-pl 1 -cl 1 -ns 2 -ar $ARITY"

all() {
  echo "Generating all tasks..."
  for i in {1..12}; do
    if [ $ARITY != 2 ] && [ $i == 8 ]; then
      continue
    fi
    #F=$DDIR'train_ar'$ARITY'_10k_task'$i.txt
    #TF=$DDIR'test_ar'$ARITY'_10k_task'$i.txt
    F=$DDIR'train_ar'$ARITY'_task'$i.txt
    TF=$DDIR'test_ar'$ARITY'_task'$i.txt
    echo Writing to $F $TF
    rm -f $F $TF
    $DCMD $ARGS -t $i -s $SIZE >> $F
    $DCMD $ARGS -t $i -s $TSIZE >> $TF
  done
}

# Run given function
$FUNC "$@"
