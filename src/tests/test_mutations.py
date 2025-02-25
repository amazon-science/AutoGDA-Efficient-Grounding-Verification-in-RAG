### Tests for metrics

## Create two equal populations.
from src.sync_data.mutation import WordReplacementMutation, WordExtensionMutation, WordDeletionMutation
from src.sync_data.mutation import RephraseMutation, APIRephrasingMutation
import sys

def test_replacement_roberta():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = WordReplacementMutation(model_str="FacebookAI/roberta-large",tokenizer_str="FacebookAI/roberta-large",
                                         n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut*len(inputs)

def test_replacement_t5():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = WordReplacementMutation(model_str="google-t5/t5-base",tokenizer_str="google-t5/t5-base",
                                         n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut*len(inputs)

def test_extension_roberta():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = WordExtensionMutation(model_str="FacebookAI/roberta-large",tokenizer_str="FacebookAI/roberta-large",
                                         n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut*len(inputs)

def test_extension_t5():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = WordExtensionMutation(model_str="google-t5/t5-base",tokenizer_str="google-t5/t5-base",
                                       n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut * len(inputs)

def test_deletion():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = WordDeletionMutation(tokenizer_str="google-t5/t5-base", n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut * len(inputs)

def test_rephrasing():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = RephraseMutation(n_mutations=n_mut)

    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut * len(inputs)

def test_llm_rephrase():
    inputs = ["Hello, this is a test.", "Yes, this one as well."]
    n_mut = 3
    myword_rep = APIRephrasingMutation(n_mutations=n_mut)
    mymutations = myword_rep.mutate(inputs)
    print(mymutations, file=sys.stderr)
    assert len(mymutations) == n_mut * len(inputs)

