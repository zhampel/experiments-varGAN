import os

# Local directory of CypherCat API
VARGAN_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(VARGAN_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')
