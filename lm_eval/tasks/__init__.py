from pprint import pprint

from . import superglue
from . import glue
from . import arc
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import race 
from . import piqa
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import webqs
from . import qa4mre
from . import translation


TASK_REGISTRY = {
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    #"stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    #"record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    
    # Order by benchmark/genre?

    "lambada": lambada.LAMBADA,
    "piqa": piqa.PiQA,

    # Science related
    "pubmedqa" : pubmedqa.Pubmed_QA,
    "sciq" : sciq.SciQ,
    #"qa4mre" : qa4mre.QA4MRE,
    "qa4mre_2011" : qa4mre.QA4MRE_2011,
    "qa4mre_2012" : qa4mre.QA4MRE_2012,
    "qa4mre_2013" : qa4mre.QA4MRE_2013,

    #"triviaqa": triviaqa.TriviaQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    # "quac": quac.QuAC, # not implemented yet
    "hellaswag": hellaswag.HellaSwag, # not implemented yet
    "openbookqa": openbookqa.OpenBookQA,
    # "sat": sat.SATAnalogies, # not implemented yet
    # "squad": squad.SQuAD, # not implemented yet
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "webqs": webqs.WebQs,
    "wsc273": wsc273.WinogradSchemaChallenge273,
    "winogrande": winogrande.Winogrande,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    # arithmetic
    "arithmetic_2da": arithmetic.Arithmetic2DPlus,
    "arithmetic_2ds": arithmetic.Arithmetic2DMinus,
    "arithmetic_3da": arithmetic.Arithmetic3DPlus,
    "arithmetic_3ds": arithmetic.Arithmetic3DMinus,
    "arithmetic_4da": arithmetic.Arithmetic4DPlus,
    "arithmetic_4ds": arithmetic.Arithmetic4DMinus,
    "arithmetic_5da": arithmetic.Arithmetic5DPlus,
    "arithmetic_5ds": arithmetic.Arithmetic5DMinus,
    "arithmetic_2dm": arithmetic.Arithmetic2DMultiplication,
    "arithmetic_1dc": arithmetic.Arithmetic1DComposite,

    # TODO Perhaps make these groups of tasks
    #   e.g. anli, arithmetic, openai_translations, harness_translations
    **translation.create_tasks_from_benchmarks(translation.selected_benchmarks)
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as e:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_dict(task_name_list):
    return {
        task_name: get_task(task_name)()
        for task_name in task_name_list
    }
