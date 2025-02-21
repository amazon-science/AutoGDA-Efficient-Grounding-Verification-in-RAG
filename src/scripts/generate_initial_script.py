## generate initial data
from src.sync_data.initial_generators import FewShotPromptingGenerator
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.utils.script_utils import get_datasets
from src.utils.data_utils import perform_query_integration

mygen = FewShotPromptingGenerator(entailment_scoring_model=EntailmentCheckModel("vectara_v2", device="cuda:0"), min_req_examples=1)
expert_qa_support_groups =['Healthcare', 'VisualArts', 'Engineering', 'Business', 'Architecture', 'Law', 'Psychology', 'Education'] #'Chemistry',

for group in expert_qa_support_groups:
    group_train, group_test = get_datasets("expertqa", group)
    group_train.df = perform_query_integration(group_train.df, mode="prepend")
    df_gen = mygen.generate_samples(group_train, samples_per_evidence=4)
    df_gen.to_csv(f"sync_data/d4_expertqa-{group}.csv", index=False)