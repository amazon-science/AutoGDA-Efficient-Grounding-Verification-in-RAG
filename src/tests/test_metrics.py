### Tests for metrics

## Create two equal populations.
from src.sync_data.population import DictPopulation
from copy import deepcopy
from src.sync_data.evaluators import NNCoverageEvaluator, FrechetInceptionDistance, ManifoldClassification
from src.sync_data.evaluators import PopulationWithEmbeddings
import numpy as np

def get_populations():
    source_samples = ["This is a difficult test.",
                      "The test was impossible to pass.",
                      "The examination was slightly harder than last year's."]

    target_population = PopulationWithEmbeddings(DictPopulation({"test": source_samples}))
    source_population = PopulationWithEmbeddings(DictPopulation({"test": deepcopy(source_samples)}))
    return source_population, target_population


def test_nn_metrics():
    source_population, target_population = get_populations()
    mymetric = NNCoverageEvaluator()
    ret = mymetric(source_population, target_population)
    assert ret["unique_nns"]["test"] == 1.0
    assert ret["min_distance"]["test"] == 0.0


def test_fid():
    source_population, target_population = get_populations()
    mymetric = FrechetInceptionDistance()
    ret = mymetric(source_population, target_population)
    print(ret)
    assert np.abs(ret["fid"]["test"]) < 0.05

def test_precision_recall():
    source_population, target_population = get_populations()
    mymetric = ManifoldClassification(nhoods=[2])
    ret = mymetric(source_population, target_population)
    print(ret)
    assert ret["precision"]["test"] == 1.0
    assert ret["recall"]["test"] == 1.0
    assert ret ["f1"]["test"] == 1.0




