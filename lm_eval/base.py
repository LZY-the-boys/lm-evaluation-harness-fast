import abc
from typing import Iterable
import numpy as np
import random
import re
import os
import json
import hashlib
from sqlitedict import SqliteDict
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lm_eval.metrics import mean, weighted_perplexity, weighted_mean, bits_per_byte
from lm_eval import utils
from abc import abstractmethod


class LM(abc.ABC):
    def __init__(self):
        self.cache_hook = CacheHook(None)

    @abstractmethod
    def loglikelihood(self, requests):
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other 
        LM calls whenever possible.

        :param requests: list
            A list of pairs (context, continuation)
            context: str
                Context string. Implementations of LM must be able to handle an 
                empty context string.
            continuation: str
                The continuation over which log likelihood will be calculated. If 
                there is a word boundary, the space should be in the continuation. 
                For example, context="hello" continuation=" world" is correct.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    @abstractmethod
    def loglikelihood_rolling(self, requests):
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementaitons
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list
            A list of strings
            string: str
                String for which we are computing per-toke  loglikelihood
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    # TODO: Add an optional max length
    @abstractmethod
    def greedy_until(self, requests):
        """Generate greedily until a stopping sequence

        :param requests: list
            A list of pairs (context, until)
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences 
                may each span across multiple tokens, or may be part of one token.
        :return: list
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook


class BaseLM(LM):

    @property
    @abstractmethod
    def eot_token_id(self):
        pass

    @property
    @abstractmethod
    def max_length(self):
        pass

    @property
    @abstractmethod
    def max_gen_toks(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def tok_encode(self, string: str): pass
    
    @abstractmethod
    def tok_decode(self, tokens: Iterable[int]): pass

    @abstractmethod
    def _model_generate(self, context, max_length, eos_token_id): pass

    @abstractmethod
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        pass

    # subclass must implement properties vocab_size, eot_token_id, max_gen_toks, batch_size, device, max_length.
    # TODO: enforce this somehow

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        for string, in tqdm(requests):
            rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )))

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)
            
            # discard is_greedy
            string_nll = [x[0] for x in string_nll]
            
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)
        
        # TODO: automatic (variable) batch size detection for vectorization
        reord = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length+1):][:-1],
                    dtype=torch.long
                ).to(self.device)
                inplen, = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad length from seq to padding_length
                inp = torch.cat([
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                ], dim=0)

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(self._model_call(batched_inps), dim=-1).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks \
                    in zip(chunk, multi_logits, inps, inplens, cont_toks_list):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen-contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return reord.get_original(res)
    
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are 
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]
        
        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str):
                until = [until]

            primary_until, = self.tok_encode(until[0])
            
            context_enc = torch.tensor([self.tok_encode(context)[self.max_gen_toks - self.max_length:]]).to(self.device)

            cont = self._model_generate(context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until)

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return reord.get_original(res)


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """
    def __init__(self):
        self.download()
        self._training_docs = None
        self._fewshot_docs = None

    def download(self):
        """Downloads the task dataset if necessary"""
        pass

    def should_decontaminate(self):
        """Whether this task supports decontamination against model training set."""
        return False

    @abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc):
        print("Override doc_to_decontamination_query with document specific decontamination query.")
        assert(False)

    @abstractmethod
    def doc_to_text(self, doc):
        pass

    @abstractmethod
    def doc_to_target(self, doc):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metric scores
        """
        pass

    @abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        pass

    def fewshot_description(self):
        import warnings
        warnings.warn(
            "`fewshot_description` will be removed in futures versions. Pass "
            "any custom descriptions to the `evaluate` function instead.",
            DeprecationWarning)
        return ""

    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]
            ) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example


class MultipleChoiceTask(Task, abc.ABC):
    def doc_to_target(self, doc):
        return " " + doc['choices'][doc['gold']]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0]
            for choice in doc['choices']
        ]

        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1. if np.argmax(results) == gold else 0.
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }
    
    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }


class PerplexityTask(Task, abc.ABC):

    def should_decontaminate(self):
        """Whether this task supports decontamination against model training set."""
        return True

    def has_training_docs(self):
        return False

    def fewshot_examples(self, k, rnd):
        assert k == 0
        return []

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        assert num_fewshot == 0
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the  "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        return ""

    def higher_is_better(self):
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_decontamination_query(self, doc):
        return doc

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc, ctx):
        assert not ctx
        req = rf.loglikelihood_rolling(self.doc_to_target(doc))
        return req

    def process_results(self, doc, results):
        loglikelihood, = results
        words = self.count_words(doc)
        bytes_ = self.count_bytes(doc)
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }

    def aggregation(self):
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    @classmethod
    def count_bytes(cls, doc):
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """ Downstream tasks with custom word boundaries should override this! """
        return len(re.split(r"\s+", doc))


def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode('utf-8')).hexdigest()


class CacheHook:
    def __init__(self, cachinglm):
        if cachinglm is None: 
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict
    
    def add_partial(self, attr, req, res):
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm, cache_db):
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr):
        def fn(requests):
            res = []
            remaining_reqs = []
            
            # figure out which ones are cached and which ones are new
            for req in requests:
                hsh = hash_args(attr, req)
                if hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None

                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)
            
            # actually run the LM on the requests that do not have cached results
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res
        return fn
    
    def get_cache_hook(self):
        return CacheHook(self)


REQUEST_RETURN_LENGTHS = {
    'loglikelihood': 2,
    'greedy_until': None,
    'loglikelihood_rolling': None,
}


class Request:
    def __init__(self, request_type, args, index=None):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError('The request type {} is not implemented!'.format(request_type))

        self.request_type = request_type
        self.args = args
        self.index = index
    
    def __iter__(self):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError('This request type does not return multiple arguments!')
        for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, i)
    
    def __getitem__(self, i):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError('This request type does not return multiple arguments!')
        return Request(self.request_type, self.args, i)
    
    def __eq__(self, other):
        return self.request_type == other.request_type and self.args == other.args and self.index == other.index

    def __repr__(self):
        return f"Req_{self.request_type}{self.args}[{self.index}]\n"


class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)
        return fn


rf = RequestFactory()
