import pandas as pd

from .config import TEST_PATH, TRAIN_PATH
from .preprocess import build_text_column


def load_train():
    return pd.read_csv(TRAIN_PATH)


def load_test():
    return pd.read_csv(TEST_PATH)