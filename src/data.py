import pandas as pd

try:
    from .config import TEST_PATH, TRAIN_PATH
except ImportError:  
    from config import TEST_PATH, TRAIN_PATH


def load_train():
    return pd.read_csv(TRAIN_PATH)


def load_test():
    return pd.read_csv(TEST_PATH)