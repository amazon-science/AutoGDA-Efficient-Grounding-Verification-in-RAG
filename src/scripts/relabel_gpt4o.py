import pandas as pd
from src.utils.script_utils import init_population_from_df, init_population_from_dump
from src.utils.script_utils import get_datasets

from src.sync_data.compute_entailments import EntailmentCheckModel
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="false"
path="sync_data"
dtype="ClaudeFewShot"
#dataset="lfqa-veri"
#dataset="summedits"
dataset="summedits"
groups=['scitldr', 'news', 'sales_call', 'sales_email', 'ectsum', 'samsum', 'podcast', 'qmsumm']
#g="news"

from contextlib import redirect_stdout, redirect_stderr, contextmanager
import os
@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            with redirect_stderr(null):
                yield

mynli = EntailmentCheckModel("gpt-4o")
#mynli = EntailmentCheckModel("tasksource")
for g in groups:
    print(g)
    synth_data_train = pd.read_csv(f"{path}/{dtype}_{dataset}-{g}.csv")
    dtrain, _ , _ = get_datasets(dataset, g)
    if "tag_0" in synth_data_train.columns:
        pop_init = init_population_from_dump(synth_data_train, use_evidence=list(dtrain.df.evidence.unique()))
    else:
        pop_init = init_population_from_df(synth_data_train, use_evidence=list(dtrain.df.evidence.unique()))
    tlist = list(pop_init.tags)
    for idx, t in tqdm(enumerate(tlist), desc="Computing initial certainties", total=len(tlist)):
        with suppress():
            sent_pairs = [[t[0], sentence] for sentence in pop_init[t]]
            scores = mynli.compute_scores(sent_pairs, show_progress=False)
            #if t[1] == 0:
            #    scores[scores > 0.5] = 0.5
            #else: # t[1] ==1
            #    scores[scores < 0.5] = 0.5
            pop_init.set_initial_prob(t, scores)
        if (idx % 40) == 39: # Save intermediate
            print(idx)
            pop_init.to_dataframe().to_csv(f"{path}/{dtype}_{dataset}-{g}_relabel_gpt4o.csv")
    pop_init.to_dataframe().to_csv(f"{path}/{dtype}_{dataset}-{g}_relabel_gpt4o.csv")