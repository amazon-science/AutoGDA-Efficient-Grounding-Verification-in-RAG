## Main Experiment

### Generating Synthetic Data
While the synthetic data can be generated in the loop, for efficiency purpose, we generate the initial synthetic data for 
all datasets first using the notebook ```notebooks\GenerateSync.ipynb```.
This will generate synthetic data files that can be loaded with the ```src.sync_data.initial_generators.FromFile``` class.

### Hyperparameter tuning
Before running the actual evaluation, we perform hyperparameter tuning using optuna.
Assign a name to the study that will later be used in other scripts to retrieve the optimal parameter configuration.
```export PYTHONPATH="."; python3 ./src/scripts/hyper_opt4.py -s <insert your own studyname> -d <dataset> -g <group> --init_model <vectara_v2, alignscore, tasksource, quantile, none>```

The ```--init_model``` flag specifies which model to use to assign probabilities to the initial scores, ```--rmiss_model`` flag specifies which model to use to assign mislabeling probabilities between subsequent mutations.
* For self-supervised data generation set ```--init_model tasksource --rmiss_model tasksource``` 
* For learning from a specific teacher model and leaving the rmiss_model as a hyperparameter, just pass ```--init_model <teacher_model>```
* Passing ```--init_model gpt-4o``` works, but it is painfully slow. Therefore, we suggest to equip the synthetic data with GPT-4o scores offline and load them directly from the file. Use ```src/scripts/relabel_gpt4o.py``` to do so. Then we implemented two versions for assigning initial token probabiities. The first one uses the plain scores from GPT-4 (```--init_model none```, none meaning they are directly read from the file). The other option is to use al quantile-normalized version of the scores, pass ```--init_model quantile```. We use this variant for the experiments in our paper.


Note that the hyperparameters may need retuning for different initmodels and datasets.
The hyper_opt4.py script also supports fixing a specific model to compute rmiss (label certainty), see argument parsing code for details.

### Evaluation
After having found good hyperparameters, we would like to evaluate the configuration on multiple generation seeds. To this
end, run the script 

```
.\src\scripts\run_multiple_seeds.sh <cuda device id> <studyname> <dataset> <group> <pinitmodel> <targetmodel>
```
where studyname is used to retrieve the hyperparameters. Otherwise the arguments are the same as above, however target model
denotes which model is finetuned. We provide a specific script for running multiple seeds for the self-supervised experiment,
see ```.\src\scripts\run_multiple_seeds_selfsupervised.sh ```, which additionally takes an rmissmodel argument, e.g. call
```
.\src\scripts\run_multiple_seeds_selfsupervised.sh 0 <self-supervised hyperparameter study> ragtruth QA tasksource tasksource tasksource
```

## Computing Baselines

The performance of the baselines can be evaluated using the script
```export PYTHONPATH="."; python3 ./src/scripts/compute_baselines.py -c <baseline_config_file> -d <dataset> -g <group> --split <val or test>```
where the config file specifies which baselines are used and their corresponding parameters.
We provide two example config files:

 * ```config_files\baseline_config.json``` (for evaluation of non-finetuned baselines and for finetuning baselines on the labeled training data, this is the upper baseline in the ablation study)
 * ```config_files\baseline_config_relabel.json``` (for finetuning the baselines on training data that was labeled by some other approach, for comparision)


## Performing the main ablation study:
The ablation study contains results that need to be collected from four three runs:

(1) No finetuned models (logged in compute baselines)

(2) The scores with few-shot synthetic data and with out full pipeline:
The runs conducted in ```.\src\scripts\run_multiple_seed.sh``` log the data that was generated, also including the initial 
selection of data generated with few-shot prompting (make sure config[skip_init_eval]=false).  We provide the script 
```src\scripts\reevaluate_test.py``` to test the data on different models. In particular we would like to reevaluate the interation [0] (few-shot synthetic data) and
iteration [N] (for total synthetic data), where N is the total number of iterations. See script for the log format of the results.

(3) Finally with random selection, the runs need to be rerun unfortunately. However, we don't do hyperparameters search again,
instead we run the seeds again, but using a slighly different configuration file which replaces the selector in ```--config_file config_files/hyperparameter_random_selector.json```
To automatically rerun this configs, use the script ```.\src\scripts\run_multiple_seeds_randomsel.sh```.

To aggregate the results from all runs, see ```notebooks/PrintMetrics.ipynb```.



## Performing ablation on r_miss
To compute the ablation study on the impact of the label certainty estimation, we provide the script
```
./src/scripts/rmiss_experiment.sh
```

## Performing ablation of selecting only a single mutation
To run the study on the effect of selecting different mutations vs only one, we provide the script
```
ablation6_reeval.sh
```


## Notebooks
The repository includes several notebooks in the folder notebooks. Here is a brief description of their purpose in the work:
* ```DatasetStatistics.ipynb```: Analyze dataset sizes and statistics.
* ```DetermineValidationSet.ipynb```: Validate that the validation set is too small of efficient Finetuning, but training models on it and reporting their performance.
* ```GenerateSync.ipynb```: Generate synthetic data for the datasets offline. This prevents that data has to be regenerated in every algorithm run.
* ```HyperparameterSeach.ipynb```: Inspect the search results of the hyperparameter searches with optuna.
* ```MutationExamples.ipynb```: Compute Illustrative examples for the behavior of the mutations.
* ```PrintMetrics.ipynb```: Collect the results from log files, and convert to LaTeX-ready tables for the paper.
* ```QualitativeResults.ipynb```: Inspect optimization curves and mutations selected (only plotting, use scripts to compute the results)
* ```RmissStudy.ipynp```: Study the effect of the model used to assess the mislabeling probabilty of the mutation (only plotting results)

Additional notebooks that extend the main ones can be found in ```/notebooks_trunk```. 

## Configuration Files
Our main algorithm Auto-GDA, implemented in ```src/sync_data/generation_loop.py``` can be configured with a configuration file.
See ```config_files/example_config.json``` for an example of how such a file can look like.
We use ```config_files/hyperparameter_base.json``` as the base setup, but the individual scripts and their call parameters make changes to this config to create the final config used to run the experiment.

