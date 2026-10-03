import importlib


def test_src_train_imports():
    module = importlib.import_module("src.train")
    assert hasattr(module, "main")
