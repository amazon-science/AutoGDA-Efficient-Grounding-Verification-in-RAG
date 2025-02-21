## Classes
import src.sync_data.initial_generators as generators
import src.sync_data.pc_mutations as pc_mutations
import src.sync_data.utility_functions as utilities
import src.sync_data.combinatorial_optimization as selectors

from copy import deepcopy

def get_data_generator(parameters: dict, base_dataset="ragtruth", group="Summary"):
    """ Get a data generator object. """
    mycls = getattr(generators, parameters["class"])
    init_params = deepcopy(parameters)
    del init_params["class"]; del init_params["n_per_evidence"]; del init_params["active"]
    init_params["dataset"] = base_dataset
    init_params["group"] = group
    return mycls(**init_params)

def get_mutation(parameters: dict, persistent_model):
    """ Construct the mutation function object """
    mymutation = getattr(pc_mutations, parameters["class"])
    init_params = deepcopy(parameters)
    del init_params["class"]
    if "active" in init_params:
        del init_params["active"]
    if persistent_model is not None:
        print("Warning, ignoring mutation.entail_model.")
        init_params["entail_model"] = persistent_model
    return mymutation(**init_params)

def get_utilites(parameters: dict, persistent_model):
    """ Construct the utility function object """
    myutil = getattr(utilities, parameters["class"])
    init_params = deepcopy(parameters)
    del init_params["class"]
    if "active" in init_params:
        del init_params["active"]
    if persistent_model is not None:
        parameters["model"] = persistent_model
    return myutil(**init_params)

def get_selector(parameters: dict, target_population: "PopulationWithEmbeddings", utility_fn=None):
    """ Construct the selector object """
    myselector = getattr(selectors, parameters["class"])
    init_params = deepcopy(parameters)
    del init_params["class"]
    if "active" in init_params:
        del init_params["active"]
        print("Warning, ignoring selector.active.")
    init_params["utility_fn"] = utility_fn
    init_params["target_population"] = target_population
    return myselector(**init_params)