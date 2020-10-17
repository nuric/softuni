# Learning Invariants through Soft Unification
Soft unification is an attention mechanism to align inputs to extract common patterns, referred to as invariants, in the data. By learning which symbols are variables the remaining part of an example is lifted into an invariant to answer the question: *if and how can a machine learn and use the idea that a symbol can take on different values?* For example, learning that Mary could be *someone* else and that person could go *somewhere* else.

## Data
The experiments are held on 5 datasets:

 - 2 toy datasets for sequence of symbols designed to test MLPs and grid of symbols for CNNs. These are generated on the fly when the `umlp.py` and `ucnn.py` training scripts are run. You can also save the generated datasets using `--data save` or `--data load` arguments.
 - The [bAbI dataset](https://research.fb.com/downloads/babi/) which can be obtained from this [direct link](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz). We use the 1k English version, for the exact task files we use, you can check the [UMN analysis notebook](analyse_umn.ipynb) and the source code for how it is loaded and processed.
 - A slightly modified version of [DeepLogic](https://github.com/nuric/deeplogic) dataset which can be generated using the `gen_dl.sh` script provided in the repository. The modification includes correct context rules to select to enable strong supervision.
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
./gen_dl.sh all 1000
```

and it will create 1000 logic programs each with one positive and one negative label for all the tasks into the `data` folder fixing predicate and constant length to 1 and noise to 2. This gives a total of 2k data points. For how to load and process the data you can check the [source for UMN model](umn.py).

## Training
The dependencies can be installed using:

```bash
pip3 install -r requirements.txt
```

For the exact environment specifications, packages used as well as hardware details, you can refer to the [environment.ipynb](environment.ipynb)

### UMLP & UCNN & URNN
The training for unification feed-forward networks, unification convolutional neural networks and unification recurrent neural networks are contained in [umlp.py](umlp.py), [ucnn.py](ucnn.py) and [urnn.py](urnn.py). For UMLP and UCNN the data generation can happen on the fly. For URNN you need get the corresponding data files as well as the ConceptNet embeddings described in the data section. To reproduce the results with different invariants and training sizes:

```python
configs = []

for script in ['umlp.py', 'ucnn.py', 'urnn.py']:
  for lr in [0.0001, 0.001, 0.01]:
    configs.append([script, '-nu', '-t', 1000, '-lr', lr])
    configs.append([script, '-nu', '-t', 50, '-lr', lr])
    for inv in range(1, 5):
      configs.append([script, '-i', inv, '-t', 1000, '-lr', lr])
      configs.append([script, '-i', inv, '-t', 50, '-lr', lr])
```

where each entry runs on a 5-fold cross-validation. You can then aggregate and analyse the results using the provided [analysis notebook](analyse_umlp_ucnn.ipynb).

You can also run each model separately and they expose different arguments which can be obtained by running with the `-h` option. For example:

```bash
python3 umlp.py -h

usage: umlp.py [-h] [--name NAME] [-l LENGTH] [-s SYMBOLS] [-i INVARIANTS] [-e EMBED] [-d] [-nu]
               [-t TRAIN_SIZE] [-g GSIZE] [-bs BATCH_SIZE] [-lr LEARNING_RATE]
               [--data {save,load}]

Run UMLP on randomly generated tasks.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name prefix for saving files etc.
  -l LENGTH, --length LENGTH
                        Fixed length of symbol sequences.
  -s SYMBOLS, --symbols SYMBOLS
                        Number of symbols.
  -i INVARIANTS, --invariants INVARIANTS
                        Number of invariants per task.
  -e EMBED, --embed EMBED
                        Embedding size.
  -d, --debug           Enable debug output.
  -nu, --nouni          Disable unification.
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        Training size per task, 0 to use everything.
  -g GSIZE, --gsize GSIZE
                        Random data tries per task.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Training batch size.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate.
  --data {save,load}    Save or load generated data.
```

### Unification Memory Networks
All the code for training and debugging including some plotting is contained in [umn.py](umn.py). To run the experiments in the paper, as well as the Iterative Memory Attention baseline:

```python
import glob

babi_tasks = glob.glob("data/tasks_1-20_v1-2/en/qa*_train.txt")
dl_tasks = glob.glob("data/deeplogic/train_1k*.txt")

configs = []

for babit in babi_tasks:
  for inv in [1, 3]:
    for runc in range(3):
      configs.append(['umn.py', babit, '-r', inv, '--runc', runc, '--name', len(configs)])
      configs.append(['umn.py', babit, '-r', inv, '--runc', runc, '--name', len(configs), '-w'])
  for runc in range(3):
    configs.append(['umn.py', babit, '-r', 3, '--runc', runc, '-t', 50, '--name', len(configs)])

for dlt in dl_tasks:
  for inv in [1, 3]:
    for runc in range(3):
      configs.append(['umn.py', dlt, '-r', inv, '--runc', runc, '--name', len(configs)])
      configs.append(['umn.py', dlt, '-r', inv, '--runc', runc, '--name', len(configs), '-w'])
  for runc in range(3):
    configs.append(['umn.py', dlt, '-r', 3, '--runc', runc, '-t', 50, '--name', len(configs)])

for dlt in dl_tasks:
  for runc in range(3):
    configs.append(['ima.py', dlt, '--runc', runc, '--name', len(configs)])
    configs.append(['ima.py', dlt, '--runc', runc, '--name', len(configs), '-w'])
```

which capture the configurations used to train the models. You can also run the script on single data files with custom options as described below:

```bash
python3 umn.py -h

usage: umn.py [-h] [--name NAME] [-r RULES] [-e EMBED] [-d] [-t TRAIN_SIZE] [-w] [--runc RUNC]
              task

Run UMN on given tasks.

positional arguments:
  task                  File that contains task train.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name prefix for saving files etc.
  -r RULES, --rules RULES
                        Number of rules in repository.
  -e EMBED, --embed EMBED
                        Embedding size.
  -d, --debug           Enable debug output.
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        Training size, 0 means use everything.
  -w, --weak            Weak supervision setting.
  --runc RUNC           Run count of the experiment, for multiple runs.
```

where the `debug` flag drops into `ipdb` after the training with a test story that produced the wrong answer or the last test story for inspection.

## FAQ

 - **Why in some parts of the source code are invariants referred to as rules?** We initially referred to invariants as rules in how humans abstract away common patterns as a rule. However, a rule relates to logic and logical constructs. Since neither the invariant structure needs to be rule-like nor the variables carry logical semantics, we decided to call them invariants.

## Built With

  - [Chainer](https://chainer.org) - deep learning framework
  - [Matplotlib](https://matplotlib.org/) - main plotting library
  - [seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
  - [NumPy](http://www.numpy.org/) - main numerical library for data vectorisation
  - [Pandas](https://pandas.pydata.org/) - helper data manipulation library
  - [jupyter](https://jupyter.org) - interactive environment to analyse data / results
