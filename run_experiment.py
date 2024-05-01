"""
The script that runs an experiment. This is the script that should be
called in HPC jobs.
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union

import torch

from _utils import print_delay
from conditions import TrainCondition, TestCondition
from loaders.load_data import rc_types as all_rc_types
from loaders.load_models import load_bert_model, load_bert_tokenizer, \
    num_layers
from test import test
from train import train

ResultList = List[Tuple[str, Optional[int], float, float]]


def _report_result(rc_type: str, alter_layer: Optional[int], accuracy: float,
                   perplexity: float, results_list: ResultList):
    """
    Helper function
    """
    if alter_layer is None:
        alter_layer_text = "No Alteration"
    else:
        alter_layer_text = "Layer {} Altered".format(alter_layer)

    print_delay("=" * 20)
    print("{} Accuracy {}: {}".format(rc_type, alter_layer_text, accuracy))
    print("{} Perplexity {}: {}".format(rc_type, alter_layer_text, perplexity))
    print_delay("=" * 20)

    results_list.append((rc_type, alter_layer, accuracy, perplexity))


def get_results_filename(model_name: str, train_condition: TrainCondition,
                         test_condition: TestCondition, dir_: Path,
                         alpha: float, k: int, glob: bool = False) -> Path:
    """
    Creates a unique CSV filename in which to save the results of the
    experiment.
    """
    max_depth = 1000
    model_name = model_name.replace("/", "_")
    train_condition = train_condition.name
    test_condition = test_condition.name

    def _get_fn(j: Union[int, str]) -> Path:
        if alpha != 4. or k != 8:
            return dir_ / "{}_{}_{}_alpha={}_k={}_results_{}.csv".format(
                model_name, train_condition, test_condition, alpha, k, j)
        return dir_ / "{}_{}_{}_results_{}.csv".format(
            model_name, train_condition, test_condition, j)

    if glob:
        return _get_fn("*")
    else:
        for i in range(max_depth):
            filename = _get_fn(i)
            if not os.path.isfile(filename):
                return filename

    raise RecursionError("There are too many results files for {} (max "
                         "limit: {})".format(_get_fn(0), max_depth))


def run_trial(model_name: str, train_condition: TrainCondition,
              test_condition: TestCondition, alpha: float, k: int,
              saved_vectors_dir: Optional[Union[str, Path]] = None,
              results_filename: Union[str, Path] = "results.csv"):
    """
    Runs one trial of the experiment.

    REQUIRED PARAMETERS

    :param model_name: The Huggingface name of the model to be tested

    :param train_condition: Determines how the verb number feature will
        be estimated. Possible options are:
            SUBJ: Estimate from the subject token(s)
            MASK: Estimate from the masked-out main verb
            ISARE: Estimate from is/are in the main verb position

    :param test_condition: Determines which tokens will be altered
        during the experiment. Possible options are:
            GLOBAL: All tokens are altered
            LOCAL: Only the subject is altered
            MASK: Only the masked-out main verb is altered
            VERB: The subject and masked-out main verb are altered

    :param alpha: The intensity of AlterRep
    :param k: The number of dimensions in the number subspace

    OPTIONAL PARAMETERS

    :param saved_vectors_dir: The directory where layer representations are
        saved
    :param results_filename: Where to save the results
    """
    tokenizer = load_bert_tokenizer(model_name)
    model = load_bert_model(model_name, alterable=True)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()

    rc_types = ["SRC", "PRC"] if test_condition == TestCondition.VERB else \
        all_rc_types
    all_results = []

    # Evaluate model without alteration
    for rc_type in rc_types:
        acc, perplexity = test(model, tokenizer, rc_type)
        _report_result(rc_type, None, acc, perplexity, all_results)

    # Evaluate model with alteration
    n_layers = num_layers(model_name)
    for layer, rc_type in itertools.product(range(n_layers), rc_types):
        number_features = train(
            saved_vectors_dir, model_name, rc_type, layer,
            train_condition=train_condition,
            num_vecs=k)

        print("Testing agreement...")
        acc, perplexity = test(model, tokenizer, rc_type, alpha=alpha,
                               alter_layer=layer,
                               test_condition=test_condition,
                               alter_dimensions=number_features)

        _report_result(rc_type, layer, acc, perplexity, all_results)

    # Save results
    with open(results_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["RCType", "LayerAltered", "Accuracy", "Perplexity"])
        for row in all_results:
            writer.writerow(row)


def run_experiment(model_name: str, train_condition: TrainCondition,
                   test_condition: TestCondition, n_trials: int,
                   alpha: float, k: int):
    """
    Runs multiple trials of an experiment.
    """
    with open("paths.json", "r") as f:
        paths_config = json.load(f)
    results_path = Path(paths_config["experiment_results_path"])
    pickle_path = Path(paths_config["pickled_bert_vectors_path"])

    for i in range(n_trials):
        print("#" * 20, "RUNNING TRIAL {}".format(i), "#" * 20, sep="\n")
        results_file = get_results_filename(model_name, train_condition,
                                            test_condition, results_path,
                                            alpha, k)
        run_trial(model_name, train_condition, test_condition, alpha, k,
                  saved_vectors_dir=pickle_path, results_filename=results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model name
    parser.add_argument("-model", "-m", type=str, default="bert-base-uncased")

    # Experimental conditions
    parser.add_argument("-nt", type=int, default=1)
    parser.add_argument("-train", type=str, default="subj")
    parser.add_argument("-test", type=str, default="global")
    parser.add_argument("-alpha", type=float, default=4.)
    parser.add_argument("-k", type=int, default=8)

    args = parser.parse_args()

    # Define train and test conditions
    train_conditions = {"subj": TrainCondition.SUBJ,
                        "mask": TrainCondition.MASK,
                        "is_are": TrainCondition.ISARE,
                        "random": TrainCondition.RANDOM}

    test_conditions = {"local": TestCondition.LOCAL,
                       "global": TestCondition.GLOBAL,
                       "verb": TestCondition.VERB,
                       "mask": TestCondition.MASK,
                       "control": TestCondition.CONTROL}

    run_experiment(args.model, train_conditions[args.train],
                   test_conditions[args.test], args.nt, args.alpha, args.k)
