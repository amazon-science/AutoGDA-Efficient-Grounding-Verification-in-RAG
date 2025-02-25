for seed in 1 2 3 4 5; do
export CUDA_VISIBLE_DEVICES=$1; export PYTHONPATH="."; python3 src/scripts/mutation_experiment6.py -s rtqa_tasksource_final2 -d ragtruth -g QA --seed ${seed}
export CUDA_VISIBLE_DEVICES=$1; PYTHONPATH="."; python3 src/scripts/reevaluate_test.py -d ragtruth -g QA -f abl_mutation-ragtruth-QA/LLMFillInTheGapsMutation_seed${seed} -m tasksource bart-large flan-t5-base --useiters 2
export CUDA_VISIBLE_DEVICES=$1; PYTHONPATH="."; python3 src/scripts/reevaluate_test.py -d ragtruth -g QA -f abl_mutation-ragtruth-QA/RephraseMutation_seed${seed} -m tasksource flan-t5-base bart-large --useiters 2
export CUDA_VISIBLE_DEVICES=$1; PYTHONPATH="."; python3 src/scripts/reevaluate_test.py -d ragtruth -g QA -f abl_mutation-ragtruth-QA/DropSentenceMutation_seed${seed} -m tasksource flan-t5-base bart-large --useiters 2
done