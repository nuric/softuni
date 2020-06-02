# Learning Invariants through Soft Unification
Soft unification is an attention mechanism to align inputs to extract common patterns, referred to as invariants, in the data. By learning which symbols are variables the remaining part of a ground example is lifted into an invariant to answer the question: *if and how can a machine learn and use the idea that a symbol can take on different values?* For example, learning that Mary could be *someone* else and that person could go *somewhere* else.

## Data
The experiments are held on 5 datasets:

 - 2 toy datasets for sequence of symbols designed to test MLPs and grid of symbols for CNNs. These are generated on the fly when the `umlp.py` and `ucnn.py` training scripts are run.
 - The [bAbI dataset](https://research.fb.com/downloads/babi/) which can be obtained from this [direct link](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz).
 - A slightly modified version of [DeepLogic](https://github.com/nuric/deeplogic) dataset which can be generated using the `gen_dl.sh` script provided in the repository.
 - And the [Sentiment Treebank dataset](https://nlp.stanford.edu/sentiment/index.html), [direct link](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip), for which we use the [ConceptNet NumberBatch](https://github.com/commonsense/conceptnet-numberbatch) word embeddings, specifically we use [version 19.08](https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz).

Except for the 2 toy datasets, the scripts expect the data to be in `data/` folder. For the downloaded datasets, you can extract the zip file into the `data/` folder.

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

for type in mlp cnn urnn; do
  echo "Running $type baseline"
  $PYTHON u$type.py base$type -i 1 -nu -t 1000
  $PYTHON u$type.py base$type -i 1 -nu -t 50

  echo "Running unification $type"
  for inv in {1..4}; do
    $PYTHON u$type.py u$type -i $inv -t 1000
    $PYTHON u$type.py u$type -i $inv -t 50
  done
done
```

You can then aggregate and analyse the results using the provided notebook: `analyse_umlp_ucnn.ipynb`

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

## FAQ

 - **Why in some parts of the source code are invariants referred to as rules?** We initially referred to invariants as rules in how humans abstract away common patterns as a rule. However, a rule relates to logic and logical constructs. Since neither the invariant structure needs to be rule-like nor the variables carry logical semantics, we decided to call them invariants.

## Built With

  - [Chainer](https://chainer.org) - deep learning framework
  - [Matplotlib](https://matplotlib.org/) - main plotting library
  - [seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
  - [NumPy](http://www.numpy.org/) - main numerical library for data vectorisation
  - [Pandas](https://pandas.pydata.org/) - helper data manipulation library
  - [jupyter](https://jupyter.org) - interactive environment to analyse data / results
