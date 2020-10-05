import datasets
import numpy as np
import random
from ..base import Dataset


class NLP_TASK(Dataset):
    NLP_PATH = None
    NLP_NAME = None

    def __init__(self):
        super().__init__()
        self._training_docs = None
        self.data = datasets.load_dataset(path=self.NLP_PATH, name=self.NLP_NAME)

    def has_training_docs(self):
        """Whether the task has a training set"""
        return True if "train" in self.data.keys() else False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True if "validation" in self.data.keys() else False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True if "test" in self.data.keys() else False

    def training_docs(self):
        # Cache training for faster few-shot.
        # If data is too large to fit in memory, override this method.
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.data["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test"]

    def fewshot_examples(self, k):
        training_docs = self.training_docs()
        n = len(training_docs)
        indices = random.sample(range(n), k)
        return [training_docs[i] for i in indices]


def simple_accuracy_metric(preds, golds):
    acc = float((np.array(preds) == np.array(golds)).mean())
    return {
        "major": acc,
        "minor": {"acc": acc},
        "higher_is_better": True,
    }


def yesno(x):
    if x:
        return 'yes'
    else:
        return 'no'
