"""
CoQA: A Conversational Question Answering Challenge
https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering 
systems. The goal of the CoQA challenge is to measure the ability of machines to 
understand a text passage and answer a series of interconnected questions that 
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/
"""
import inspect
import transformers.data.metrics.squad_metrics as squad_metrics
import lm_eval.datasets.coqa.coqa
from lm_eval.base import PromptSourceTask, Task, rf, mean
from itertools import zip_longest


_CITATION = """
@misc{reddy2018coqa,
    title={CoQA: A Conversational Question Answering Challenge},
    author={Siva Reddy and Danqi Chen and Christopher D. Manning},
    year={2018},
    eprint={1808.07042},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class CoQA(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "coqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        pass

    # @classmethod
    # def get_answers(cls, doc, turn_id):
    #     # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    #     answers = []
    #     answer_forturn = doc["answers"]["input_text"][turn_id - 1]
    #     answers.append(answer_forturn)
    #     additional_answers = doc.get("additional_answers")
    #     if additional_answers:
    #         for key in additional_answers:
    #             additional_answer_for_turn = additional_answers[key]["input_text"][
    #                 turn_id - 1
    #             ]
    #             if additional_answer_for_turn.lower() not in map(str.lower, answers):
    #                 answers.append(additional_answer_for_turn)
    #     return answers

    @staticmethod
    def compute_scores(gold_list, pred):
        # tests for exact match and on the normalised answer (compute_exact)
        # test for overlap (compute_f1)
        f1_sum = 0.0
        em_sum = 0.0
        if len(gold_list) > 1:
            for i in range(len(gold_list)):
                gold_answers = gold_list[0:i] + gold_list[i + 1 :]
                # predictions compared against (n) golds and take maximum
                em_sum += max(
                    squad_metrics.compute_exact(a, pred) for a in gold_answers
                )
                f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
        else:
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)

        return {
            "em": em_sum / max(1, len(gold_list)),
            "f1": f1_sum / max(1, len(gold_list)),
        }

    def stopping_criteria(self):
        return "\n\n"

    # def construct_requests(self, doc, ctx):
    #     """Uses RequestFactory to construct Requests and returns an iterable of
    #     Requests which will be sent to the LM.

    #     :param doc:
    #         The document as returned from training_docs, validation_docs, or test_docs.
    #     :param ctx: str
    #         The context string, generated by fewshot_context. This includes the natural
    #         language description, as well as the few shot examples, and the question
    #         part of the document for `doc`.
    #     """
    #     return cont_request

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        target = self.doc_to_target(doc).strip()
        pred = results[0].strip().split("\n")[0]
        scores = self.compute_scores([target], pred)

        out = {
            "f1": scores["f1"],
            "em": scores["em"],
        }

        if self.save_examples:
            example = {
                "f1": scores["f1"],
                "em": scores["em"],
            }
            return out, example
        return out

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }
