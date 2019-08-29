# Learning Invariants through Soft Unification
Soft unification is an attention mechanism to align inputs to extract common patterns, referred to as invariants, in the data. By learning which symbols are variables the remaining part of a ground example is lifted into an invariant to answer the question: *if and how can a machine learn and use the idea that a symbol can take on different values?* For example, learning that Mary could be *someone* else and that person could go *somewhere* else.

## Data
The experiments are held on 4 datasets:

 - 2 toy datasets for sequence of symbols designed to test MLPs and grid of symbols for CNNs.
 - The [bAbI dataset](https://research.fb.com/downloads/babi/) which can be obtained from this [direct link](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz).
 - And a slightly modified version of [DeepLogic](https://github.com/nuric/deeplogic) dataset which can be generated using the `gen_dl.sh` script provided in the repository.

To generate the logical reasoning tasks individually with different parameters:

```bash
python3 gen_logic.py

usage: gen_logic.py [-h] [-t TASK] [-s SIZE] [-ns NOISE_SIZE]
                   [-cl CONSTANT_LENGTH] [-vl VARIABLE_LENGTH]
                   [-pl PREDICATE_LENGTH] [-ar ARITY] [--nstep NSTEP]

Generate logic program data.

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  The task to generate.
  -s SIZE, --size SIZE  Number of programs to generate.
  -ns NOISE_SIZE, --noise_size NOISE_SIZE
                        Size of added noise rules.
  -cl CONSTANT_LENGTH, --constant_length CONSTANT_LENGTH
                        Length of constants.
  -vl VARIABLE_LENGTH, --variable_length VARIABLE_LENGTH
                        Length of variables.
  -pl PREDICATE_LENGTH, --predicate_length PREDICATE_LENGTH
                        Length of predicates.
  -ar ARITY, --arity ARITY
                        Arity.
  --nstep NSTEP         Generate nstep deduction programs.
```

or use the provided wrapper script:

```bash
mkdir -p data/deeplogic
./gen_dl.sh all [SIZE:1000|10000] [ARITY:1|2]
```

and it will create all the tasks into the `data` folder fixing predicate and constant length to 1 and noise to 2.

## Training
The dependencies can be installed using:

```bash
pip3 install --no-cache-dir --upgrade -r requirements.txt
```

### UMLP & UCNN
The training for unification feed-forward networks and unification convolutional neural networks are contained in `umlp.py` and `ucnn.py` with all the data generation and training done together. To reproduce the results with different invariants and training sizes:

```bash
PYTHON=python3
SCRIPT=umlp.py
CMD="$PYTHON $SCRIPT"

for type in mlp cnn; do
  echo "Running $type baseline"
  $PYTHON u$type.py base$type -i 1 -nu
  $PYTHON u$type.py base$type -i 1 -nu -t 50

  echo "Running unification $type"
  for inv in {1..4}; do
    $PYTHON u$type.py u$type -i $inv
    $PYTHON u$type.py u$type -i $inv -t 50
  done
done
```

To aggregate the log files into a csv file, you can use the utils provided:

```bash
for type in umlp ucnn; do
  python3 utils/umlp_ucnn_process_logs.py -hd $type_result/*_log.json > $type_results.csv
  python3 utils/umlp_ucnn_process_logs.py $type_result/*_log.json >> $type_results.csv
done
```

### Unification Memory Networks
All the code for training and debugging including plotting is contained in `umn.py`. After the data is downloaded / generated:

```bash
python3 umn.py -h

usage: umn.py [-h] [-r RULES] [-e EMBED] [-d] [-t TSIZE] [-s STRONG] task name

Run UMN on given tasks.

positional arguments:
  task                  File that contains task train.
  name                  Name prefix for saving files etc.

optional arguments:
  -h, --help            show this help message and exit
  -r RULES, --rules RULES
                        Number of rules in repository.
  -e EMBED, --embed EMBED
                        Embedding size.
  -d, --debug           Enable debug output.
  -t TSIZE, --tsize TSIZE
                        Training size, 0 means use everything.
  -s STRONG, --strong STRONG
                        Strong supervision ratio.
```

where `task` is the task data file and `name` is the name of the file for auxiliary output files such as log and model weights:

```bash
python3 umn.py data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt qa01
```

The `debug` flag drops into `ipdb` after the training with a test story that produced the wrong answer or the last test story for inspection.
