# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno

class SQuAD(HFTask):
    DATASET_PATH = "squad_v2"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.data["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def fewshot_description(self):
        # TODO: redo description
        return "Title: The_Title_of_It\n\nBackground: A text passage as background to answer the question with.\n\nQ: Question about the passage.\n\nA: Answer."

    def doc_to_text(self, doc):
        return 'Title: ' + doc['title'] + '\n\n' + 'Background: ' + doc['context'] + '\n\n' + 'Q: ' + doc['question'] + '\n\n' + 'A: '

    def doc_to_target(self, doc):
        answer_list = doc['answers']['text']
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = 'unanswerable'
        return answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')