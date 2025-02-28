## Implement probabilistically correct mutations.

import abc
from abc import ABC
import typing as tp
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import FSMTTokenizer, FSMTForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from src.sync_data.population import Population
import torch
from tqdm import tqdm
import os
from src.utils.bedrock_utils import get_bedrock_batch_response
from multiprocessing import Pool
from sentence_transformers import CrossEncoder
from src.sync_data.compute_entailments import EntailmentCheckModel, split_sentences
from random import sample
from src.utils.script_utils import suppress_stdout

def repeat_sorted(n_max, n_repeats):
    """ return list [0,0,0, 1,1,1, ... n_max, n_max, n_max], where
    there are n_repeats of each number (3 in this example)"""
    return sum([[i] * n_repeats for i in range(n_max)], [])


class ProbCorrectMutation(ABC):
    """ Interface for probablistically correct data augmentation (PCDA). """
    def __init__(self, device, n_mutations, nli_model=None, miss_prob=0.05, check_against="parent"):
        """ nli_model: the NLI model to perform entailment checks.
            miss_prob: Fallback probabilities that this mutation will flip the label, used in case nli_model==None.
            check_against: "parent" or "evidence". What to use as evidence to compute entailment scores for this mutation.
                "parent" means that the mutation is compared to the sample it was generated from.
                "evidence" means that the entailment probability is computed directly from the evidence.
        """
        self.n_mutations = n_mutations
        self.device = device
        self.miss_prob = miss_prob
        self.check_against = check_against

        if isinstance(nli_model, str):
            self.entail_model_identifier = nli_model
            self.nli_model = EntailmentCheckModel(nli_model, device=self.device)
        else:
            self.nli_model = nli_model
            if self.nli_model is not None:
                if self.nli_model.noise > 0.0:  # A non-standard nli model, do not use cache for scores.
                    self.entail_model_identifier = "custom"
                else:
                    self.entail_model_identifier = nli_model.model_name_str
            else:
                self.entail_model_identifier = "custom"
    def mutate(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], np.ndarray]:
        """ Our mutation interface. Mutates the samples in inputs and creates self.n_mutations offspring.
            :param: inputs: A list of input samples.
            :return: 1. A mutated list of input samples. The length of output list is len(inputs)*self.n_mutations.
                     2. Additionally return a list or ndarray with the ids of the original samples that the offspring was
                     created from and
        """
        raise NotImplementedError()

    def mutate_all_tags(self, input_population: Population, return_raw_rmiss=False):
        """ Mutate tags in the population. Return a new population with mutated samples.
            This is not an in-place operation, i.e., a new Population object will be created.
        """
        new_population = Population()
        rmiss_dict = {}
        for tag in tqdm(input_population.tags, total=len(input_population.tags)):
            with suppress_stdout():
                samples, parent_ids = self.mutate(input_population[tag].tolist(), tag)
                if self.nli_model is not None:
                    pmisslabel = self._compute_rmiss(input_population[tag].tolist(), samples, parent_ids, tag=tag)
                else:
                    pmisslabel = np.ones(len(samples))*self.miss_prob
                rmiss_dict[tag] = pmisslabel
                if self.check_against == "parent":
                    p_org = input_population.get_initial_prob(tag)
                    p_org_update = p_org[parent_ids]
                    pagree_prior = input_population.get_agreement(tag)[parent_ids]  # Update misslabel probabilities.
                    pagree_update = (1.0-pmisslabel)*pagree_prior + pmisslabel*(1.0-pagree_prior)
                else: ## check_against == "evidence":
                    p_org_update = pmisslabel
                    pagree_update = np.ones(len(samples))
                new_population[tag] = samples, parent_ids, pagree_update, p_org_update
        if return_raw_rmiss:
            return new_population, rmiss_dict
        else:
            return new_population

    def num_per_sample(self):
        """ Return number of mutated samples that are expected to be returned per input for this mutation. """
        return self.n_mutations

    def _compute_rmiss(self, inputs, modified_samples, back_references, tag, show_progress=False):
        """ inputs: list of input sequences.
            modified samples: list of modified seqences.
            back_references: list of back references into the inputs. len(back_references) == len(modified_samples).
                The values in back_references should be valid indices for inputs.
            Compute the probabilities of non-entailment (opposite of p_agree).
        """
        if self.check_against == "evidence":
            sentence_pairs = [[tag[0], sample] for sample in modified_samples]
            res = self.nli_model.compute_scores(sentence_pairs, show_progress=show_progress)
            if tag[1] == 1:
                res[res < 0.5] = 0.5
            else:  # tag[1] == 0
                res[res > 0.5] = 0.5
            return res
        else: ## "parent"
            if tag[1] == 1:
                sentence_pairs = [[inputs[bref], sample] for sample, bref in zip(modified_samples, back_references)]
            else: #tag[1] == 0:
                sentence_pairs = [[sample, inputs[bref]] for sample, bref in zip(modified_samples, back_references)]
            res = self.nli_model.compute_scores(sentence_pairs, show_progress=show_progress)
            res[res < 0.5] = 0.5
            return 1.0-res

def extract_tag(llm_response: str, tag_name: str) -> str:
    pos_start = llm_response.find(f'<{tag_name}>')
    pos_end = llm_response.find(f'</{tag_name}>')
    if pos_start  == -1 or pos_end == -1:
        print("Invalid response:", llm_response)
        return None

    return llm_response[pos_start+len(tag_name)+2:pos_end]

class APIModelPCMutation(ProbCorrectMutation):
    """ Call an API model with a prompt to create mutations.
        Note that this is an abstract class. You have to specify a prompt to implement this class.
        The output will generally be expected between tags <answer #>Answer<Answer 1>.
        The final prompt will be created by formating the prompt with (n_mutations, n_mutation-1) and appending the doc.
    """

    def __init__(self, device="cuda", n_mutations=3, model_name="claude3-haiku", batch_size=10,
                 temperature=0.0, n_threads=6, entail_model='tasksource',
                 miss_prob=0.05, check_against="parent"):
        os.environ["AWS_REGION_NAME"] = 'us-east-1'
        if model_name == 'claude3-sonnet':
            self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        elif model_name == 'claude3-haiku':
            self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        elif model_name in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
            self.model_id = model_name
        else:
            raise ValueError(f"Unrecognized model name: {model_name}")

        super().__init__(device, n_mutations, entail_model, miss_prob, check_against)
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_threads = n_threads
        self.miss_prob = miss_prob

    def _generate_prompt_request(self, documents: tp.List[str], tag) -> tp.List[str]:
        """
        Format the LLM prompts for this batch.
        The
        """

        generate_prompt = lambda prefix, doc: f"""
            {prefix} The document is: 
             <document>{doc}</document>
             Assistant: """
        prompts = [
            generate_prompt(self.get_base_prompt().format(self.n_mutations, self.n_mutations - 1), doc) for doc in
            documents
        ]
        # print(prompts[0])
        return prompts

    @abc.abstractmethod
    def get_base_prompt(self):
        raise NotImplementedError("Not implemented in this abstract class.")

    def mutate(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        ret = mutate_thread([self, inputs, tag, 0])
        return ret[1], ret[2]

    def mutate_all_tags(self, input_population: Population, return_raw_rmiss=False):
        """ Mutate the samples in each tag and update the p_agree values. """
        pop = Population()
        print(f"Using {self.n_threads} threads.")
        tasklist = []

        if self.n_threads > 1:
            for idx, tag in tqdm(enumerate(input_population.tags), total=len(input_population.tags)):
                tasklist.append((self, input_population[tag].tolist(), tag, idx))

            with Pool(self.n_threads) as p:
                res_array = p.map(mutate_thread, tasklist)
        else:
            res_array = []
            for idx, tag in tqdm(enumerate(input_population.tags), total=len(input_population.tags)):
                res_array.append(mutate_thread((self, input_population[tag].tolist(), tag, idx)))
        rmiss_dict = {}
        for k, v, r in res_array:
            if self.nli_model is not None:
                pmiss = self._compute_rmiss(input_population[k].tolist(), v, r, tag=k, show_progress=True)
            else:
                pmiss = np.ones(len(r))*self.miss_prob
            rmiss_dict[k] = pmiss
            #print(pmiss)
            if self.check_against == "parent":
                p_org_update = input_population.get_initial_prob(k)[r]
                pagree_prior = input_population.get_agreement(k)[r]  # Update misslabel probabilities.
                pagree_update = (1.0 - pmiss) * pagree_prior + pmiss * (1.0 - pagree_prior)
            else: # check_against == "evidence"
                pagree_update = np.ones(len(v))
                p_org_update = pmiss
            pop[k] = v, r, pagree_update, p_org_update
        if return_raw_rmiss:
            return pop, rmiss_dict
        else:
            return pop


def mutate_thread(arglist):
    """
        Performs the LLM call and returns tuple (tag, outputs, references)
    """
    mutation_model, inputs, tag, idx = arglist
    inputs = np.array(inputs)
    output_list = []
    back_ref_list = []
    processed_inputs = 0
    for batch_start in range(0, len(inputs), mutation_model.batch_size):
        doc_batch = inputs[batch_start:batch_start + mutation_model.batch_size]
        prompts = mutation_model._generate_prompt_request(doc_batch, tag)
        ## Resubmit query until valid response was received.
        invalid = True
        while invalid:
            valid_answers = []
            invalid = False
            llm_responses = get_bedrock_batch_response(prompts, model=mutation_model.model_id,
                                                       temperature=mutation_model.temperature)

            ## Parse responses
            for res in llm_responses:
                for summ_id in range(mutation_model.n_mutations):
                    claim = extract_tag(res, f'answer {summ_id}')
                    if claim is not None:
                        valid_answers.append(claim.strip("\n"))
                    else:
                        print("invalid response received. Retrying.")
                        invalid = True
        output_list.extend(valid_answers)
        for i in range(mutation_model.n_multiply):
            back_ref_list.append(processed_inputs+np.array(repeat_sorted(len(doc_batch), mutation_model.n_mutations)))
        processed_inputs += len(doc_batch)

    back_refs = np.concatenate(back_ref_list, axis=0)
    if idx % 10 == 0:
        print("Done at index ", idx)
    return (tag, output_list, back_refs)


class LLMFillInTheGapsMutation(APIModelPCMutation):
    """ Mask parts of the sentence and let them be replaced by an LLM. """

    def __init__(self, device="cuda", n_mutations=3, model_name="claude3-haiku",
                 batch_size=10, temperature=1.0, mask_output_perc=0.2, connected=True, n_multiply=1,
                 n_threads=6, preserve_meaning=False, miss_prob=0.05, entail_model="tasksource"):
        super().__init__(device=device, n_mutations=n_mutations, model_name=model_name,
                         batch_size=batch_size, temperature=temperature, n_threads=n_threads,
                         miss_prob=miss_prob, entail_model=entail_model)
        """ Arguments:
            connected: Whether the masked words should be next to each other or not.
        """
        self.mask_output_perc = mask_output_perc
        self.connected = connected
        self.n_multiply = n_multiply
        self.preserve_meaning = preserve_meaning

    def _generate_prompt_request(self, documents: tp.List[str], tag) -> tp.List[str]:
        """
        Format the LLM prompts for this batch.
        The
        """
        generate_prompt = lambda prefix, doc: f"""
                    {prefix} The document is: 
                     <document>{doc}</document>
                     Assistant: """

        generate_prompt_org = lambda prefix, doc_org, doc_gaps: f"""
                            {prefix} You will now see the original document, but you will have to generate different versions that preserve the meaning by filling the gaps.
                             Here is the original:
                             <document>{doc_org}</document>
                             The document including the gaps is:
                             <document>{doc_gaps}</document>
                             Assistant: """

        prompts = []
        for k in range(self.n_multiply):
            # Split and Mask
            for i in range(len(documents)):
                parts = documents[i].split(" ")
                num_toks = len(parts)
                num_mask = min(int(len(parts) * self.mask_output_perc + 1.0), len(parts))
                if self.connected:
                    res = np.random.randint(num_toks - num_mask + 1)
                    parts[res:res + num_mask] = ["_"] * num_mask
                else:
                    res = np.random.randint(num_toks, size=num_mask)
                    for i in range(num_mask):
                        parts[res[i]] = "_"
                document_gaps = " ".join(parts)
                if self.preserve_meaning:
                    prompts.append(
                        generate_prompt_org(self.get_base_prompt().format(self.n_mutations, self.n_mutations - 1),
                                            documents[i], document_gaps))
                else:
                    prompts.append(
                        generate_prompt(self.get_base_prompt().format(self.n_mutations, self.n_mutations - 1),
                                        document_gaps))
        return prompts

    def get_base_prompt(self):
        return """ Your task is to fill in the gaps in a document indicated with "_" with additional details. 
            If there is no gaps, please output the input text. The number of "_" indicates the approximate number of words
            that should be filled into each gap. While slight deviations (e.g., one word more or less) are permissible,
            the filled in text should respect the length indicated through the number of "_". **Do not change the text outside the gaps and do not include gaps in the final output.**
            You will generate {} semantically different completions of the document. Each completed document is identified with an integer from 0 to {}. 
            The document with the blanks filled must be wrapped in <answer #></answer #> tags, where # is replaced with the id of the filled-in document.
        """
    def num_per_sample(self):
        """ Return number of mutated samples that are expected to be returned per input for this mutation. """
        return self.n_mutations * self.n_multiply


class LLMSynonymeReplacementMutation(APIModelPCMutation):
    """ Mask parts of the sentence and let them be replaced by an LLM. """

    def __init__(self, device="cuda", n_mutations=3, model_name="claude3-haiku",
                 batch_size=10, temperature=1.0, replace_words=3,
                 n_threads=6, miss_prob=0.05, entail_model="tasksource"):
        super().__init__(device=device, n_mutations=n_mutations, model_name=model_name,
                         batch_size=batch_size, temperature=temperature, n_threads=n_threads,
                         miss_prob=miss_prob, entail_model=entail_model)
        """ Arguments:
            connected: Whether the masked words should be next to each other or not.
        """
        self.replace_words = replace_words
        self.n_multiply = 1

    def _generate_prompt_request(self, documents: tp.List[str], tag) -> tp.List[str]:
        """
        Format the LLM prompts for this batch.
        The
        """
        generate_prompt = lambda prefix, doc: f"""
                    {prefix} The document is: 
                     <document>{doc}</document>
                     Assistant: """

        prompts = []
        # Split and Mask
        for i in range(len(documents)):
            prompts.append(
                generate_prompt(self.get_base_prompt().format(self.replace_words, self.replace_words, self.n_mutations, self.n_mutations - 1), documents[i]))
        return prompts


    def get_base_prompt(self):
        return """ Your task is to replace words in a given document with synonyms. Specifically, you will be given a document \
and it is your task to select {} words and replace these {} words by synonyms that preserve the meaning of the sentence.
Do not only change nouns, but also pronouns, verbs and adjectives! **Do not change the text besides the words that you selected for replacement.**
You will provide {} answers to this task. The answers contain different versions of the document. You can select a different set of words to be \
replaced in each answer, or replace the same words by different synonyms. Each answer document is identified by an integer from 0 to {}. 
The document with the replacements must be wrapped in <answer #></answer #> tags, where # is replaced with the id of the modified answer document."""

    def num_per_sample(self):
        """ Return number of mutated samples that are expected to be returned per input for this mutation. """
        return self.n_mutations * self.n_multiply


class LLMInformationIntroductionMutation(APIModelPCMutation):
    """ Mask parts of the sentence and let them be replaced by an LLM. """

    def __init__(self, device="cuda", n_mutations=3, model_name="claude3-haiku",
                 batch_size=10, temperature=1.0, n_threads=6, miss_prob=0.05,
                 entail_model="tasksource", check_against="evidence", n_multiply=1):
        super().__init__(device=device, n_mutations=n_mutations, model_name=model_name,
                         batch_size=batch_size, temperature=temperature, n_threads=n_threads,
                         miss_prob=miss_prob, entail_model=entail_model, check_against=check_against)
        """ Arguments:
            connected: Whether the masked words should be next to each other or not.
        """
        self.n_multiply = n_multiply

    def word_overlap_metric(self, claim, sentence):
        """ compute how many of the words in a sentence are already present in a claim.
            This metric will be used to select the sentences with the fewest overlap for integration.
        """
        blacklist = ["Briefly answer the following question:",
                     "In case the passages do not contain the necessary information to answer the question",
                     "You are given the question:",
                     "Here is some information related to the question:"]
        for s in blacklist:
            if s in sentence:
                return 1.0
        ## not in blacklist
        words_sentence = set(sentence.split(" "))
        words_claim = set(claim.split(" "))
        return len(words_claim.intersection(words_sentence))/len(words_sentence)

    def _generate_prompt_request(self, documents: tp.List[str], tag) -> tp.List[str]:
        """
        Format the LLM prompts for this batch.
        The
        """
        prompts = []
        evidence_pieces = split_sentences(tag[0])
        # Split and Mask
        for i in range(len(documents)):
            sentence_stack = ""
            overlap_scores = np.array([self.word_overlap_metric(documents[i], piece) for piece in evidence_pieces])
            #print(overlap_scores)
            sentences_intro = np.argsort(overlap_scores)[:self.n_mutations]
            if len(sentences_intro) < self.n_mutations: ## Fewer sentences? repeat sentences.
                sentences_intro = np.tile(sentences_intro, self.n_mutations)[:self.n_mutations]
            for k in range(self.n_mutations):
                sentence_stack += f"<sentence {k}>{evidence_pieces[sentences_intro[k]]}<\sentence {k}>\n"
            prompts.append(self.get_base_prompt().format(documents[i], self.n_mutations, self.n_mutations-1,
                                                         sentence_stack, self.n_mutations, self.n_mutations-1))
        return prompts


    def get_base_prompt(self):
        return """ Your task is to add additional information to a paragraph of text. You are given the following paragraph in <paragraph><\paragraph> tags:
<paragraph>{}</paragraph>
Now you are given {} sentences or sentence pieces in <sentence #></sentence #> tags, where the # is replaced by a sentence number from 0 to {}. Here are the sentences: \n
{}\n
Your task is to create {} modified versions of the paragraph which integrating one of the sentences given in the paragraph at a sensible place, while keeping everything else as is. \
If the information given in the sentence is already present in the paragraph the paragraph can stay untouched. If parts of the information are already present, integrate the remainder. \
You will wrap your answers in <answer #></answer #> tags, where answer is replaced by a number from 0 to {}. \
For instance, <answer 0>...</answer 0> will contain the paragraph integrating the information present in sentence 0. <answer 1>...</answer 1> contains the paraphraph with the information from sentence 1, etc. 
Assistent: """

    def num_per_sample(self):
        """ Return number of mutated samples that are expected to be returned per input for this mutation. """
        return self.n_mutations * self.n_multiply

class RephraseMutation(ProbCorrectMutation):
    """ Use a local paraphrasing model to do the mutation step. """
    def __init__(self,
                 device="cuda",
                 n_mutations=3,
                 num_beam_groups=1,
                 repetition_penalty=10.0,
                 diversity_penalty=0.0,
                 no_repeat_ngram_size=5,
                 temperature=1.0,
                 max_length=128,
                 do_sample=True,
                 miss_prob=0.05, entail_model="tasksource"
                 ):
        super().__init__(device, n_mutations, entail_model, miss_prob)
        self.num_beam_groups = num_beam_groups
        self.repetition_penalty = repetition_penalty
        self.diversity_penalty = diversity_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.temperature = temperature
        self.max_length = max_length
        self.do_sample = do_sample
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(self.device)

    def mutate(self, inputs, tag=None):

        input_list = list([f'paraphrase: {question}' for question in inputs]) #add prompts
        input_ids = self.tokenizer(
            input_list,
            return_tensors="pt", padding="longest",
            max_length=self.max_length,
            truncation=True,
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=self.n_mutations, no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_beams=self.n_mutations, num_beam_groups=self.num_beam_groups,
            max_length=self.max_length, diversity_penalty=self.diversity_penalty,
            do_sample=self.do_sample
        )

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res, repeat_sorted(len(inputs), self.n_mutations)


class DropSentenceMutation(ProbCorrectMutation):
    """ Drops sentences from the claim. """
    def __init__(self, device="cuda", miss_prob=0.05, entail_model="tasksource", n_mutations=3, num_dropped=1):
        super().__init__(device, n_mutations, entail_model, miss_prob)
        self.num_dropped = num_dropped

    def mutate(self, inputs, tag=None):
        ret_list = []
        for inp in inputs:
            sentence_list = split_sentences(inp)
            num_remain = max(len(sentence_list)-self.num_dropped, 1) # Make sure at least one sentence remains.
            num_drop = len(sentence_list) - num_remain
            for k in range(self.n_mutations):
                dropped = np.random.permutation(len(sentence_list))[:num_drop].tolist()
                new_sample = " ".join([sentence_list[s] for s in range(len(sentence_list)) if s not in dropped])
                ret_list.append(new_sample)
        return ret_list, repeat_sorted(len(inputs), self.n_mutations)


class MaskedLMMutationBase(ProbCorrectMutation):
    """ Use a masked language model to create mutations by replacing a token a sentence.
        This class serves as the base class to WordReplacementMutation and TokenExtensionMutation.
        Note that this may not preserve entailment. Entailment needs to be checked in a dedicated step.
        We can use classical MaskedLMs like "FacebookAI/roberta-large", "distilbert/distilroberta-base",
        but also provide support for the T5-style Seq2Seq models (the class will be automatically determined by whether
        "t5" is in the name)
        for instance use "google-t5/t5-base" as model_str. Note that the replacement procedure might be specific to this
        model, so functionality with other models cannot be guaranteed.
    """
    def __init__(self, device='cuda', n_mutations=3, model_str="FacebookAI/roberta-large",
                 tokenizer_str="FacebookAI/roberta-large", max_length=512, consider_top_k=200,
                 n_multiply = 1, miss_prob=0.05, entail_model="tasksource", rerank_similarity=False):
        """
            :param: max_length: maximum length of the model used.
            :param: consider_top_k: before sampling the top k most likely tokens of the entire vocab of use_vocab are
                selected. Then we sample the required n_mutations form this pool randomly.
            :param: model_str: currently supported models are "FacebookAI/roberta-large", "distilbert/distilroberta-base", "google-t5/t5-base"
            :param: use_vocab: vocabulary considered for replacement. If None, the entire vocabulary will be used.
        """
        super().__init__(device, n_mutations, entail_model, miss_prob)
        # t5_style indicates whether a MaskedLM or Seq2Seq Architecture is used.
        self.t5_style = ("t5" in model_str)

        if not self.t5_style: #Masked LM
            self.model = AutoModelForMaskedLM.from_pretrained(model_str).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        else: #Seq2Seq LM
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(device)
        self.max_length = max_length
        self.temperature = 1.0
        self.consider_top_k = consider_top_k
        self.top_k = consider_top_k  # number of topk tokens to consider
        self.batch_size = 64
        self.n_multiply = n_multiply
        self.rerank_similarity = rerank_similarity

    def _forward_t5_replacements(self, input_ids):
        """ Compute the token probabilities for replacement using the t5 model. ONLY use with a T5 model.
            The sequence to sequence architecture first expects the mask token as output sequence e.g.
            <extra_id_0> and will then output the completion text (which may consist of several tokens).
            However, as we are only interested in a single token, we only consider the token probabilities output after
            this on.
            See https://huggingface.co/docs/transformers/en/model_doc/t5#training for more details.
        """
        inputs_embeds = self.model.encoder.embed_tokens(input_ids.cuda()).detach()
        mask_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        labels = -100 * torch.ones(inputs_embeds.shape[0], 2, dtype=torch.long)
        labels[:, 0] = mask_id
        logits = self.model(inputs_embeds=inputs_embeds.cuda(), labels=labels.long().cuda()).logits[:, 1, :]
        return logits

    def get_embeddings(self, token_ids):
        if self.t5_style:
            return self.model.encoder.embed_tokens(token_ids)
        elif "roberta" in self.model._modules:
            return self.model.roberta.embeddings.word_embeddings(token_ids)
        else:
            raise ValueError("Cannot compute embeddings for unknown model architecture.")
    def _get_topk_replacements(self, ret_masking, replace_ids, org_ids=None):
        """
            Return the extended inputs, each with k replacements for the originally masked tokens.
            ret_masking: Masked inputs
            replace_ids: The index in each input that was masked.
        """
        if self.t5_style:
            my_masked_lm_probs = self._forward_t5_replacements(ret_masking["input_ids"])
        else:
            my_masked_lm_probs = self.model(input_ids=ret_masking["input_ids"].to(self.device),
                                            attention_mask=ret_masking["attention_mask"].to(self.device)).logits
            my_masked_lm_probs = my_masked_lm_probs[torch.arange(len(replace_ids)), replace_ids] # only select relevant probabilities

        # my_masked_lm_probs is of shape Batch x Vocab and now contains the logits for each token.

        # Find topk indices
        _, indices = torch.topk(my_masked_lm_probs, self.top_k, dim=-1)

        if self.rerank_similarity:
            org_tokens = org_ids[torch.arange(len(replace_ids)), replace_ids]
            org_embeddings = self.get_embeddings(org_tokens.to(self.device))
            org_embeddings = org_embeddings / torch.norm(org_embeddings, dim=-1, keepdim=True)
            replace_embeddings = self.get_embeddings(indices.to(self.device))
            replace_embeddings = replace_embeddings / torch.norm(replace_embeddings, dim=-1, keepdim=True)
            distmat = torch.cdist(org_embeddings.unsqueeze(1), replace_embeddings) # [B, 1, top_k]
            new_order = distmat.squeeze(1).argsort(dim=1)
            indices = torch.gather(indices, 1, new_order)
        indices = indices.cpu()

        ### Assemble output sequences (sampling randomly from the topk)
        outputs = ret_masking["input_ids"].expand(self.n_mutations, *ret_masking["input_ids"].shape).clone()
        # print(indices[:, :, torch.randperm(indices.shape[2])[:n_outputs]].shape)
        for i in range(len(ret_masking["input_ids"])):
            outputs[:, i, replace_ids[i]] = indices[i, torch.randperm(indices.shape[1])[:self.n_mutations]]
        return self.tokenizer.batch_decode(outputs.reshape(-1, ret_masking["input_ids"].shape[-1]), skip_special_tokens=True)

    def num_per_sample(self):
        """ Return number of mutated samples that are expected to be returned per input for this mutation. """
        return self.n_mutations * self.n_multiply

class WordReplacementMutation(MaskedLMMutationBase):
    def _do_masking_replace(self, sequences):
        """ Mask one random token in each sequence (internal helper function)
            Masks one token in each input sequence randomly.
            Return the masked input (dict), the token_idx of the masked token for each input, and the original input_ids
        """
        ret = self.tokenizer(sequences, padding=True, return_tensors="pt", truncation=True)
        org_ids = ret["input_ids"].clone()
        seq_len = torch.sum(ret["attention_mask"], dim=1)
        mask_prob = ret["attention_mask"].clone()
        mask_prob[torch.arange(len(mask_prob)), seq_len - 1] = 0
        mask_prob[:, 0] = 0
        norm_attn_mask = ret["attention_mask"] / seq_len.reshape(-1, 1)
        mask_seq_token = torch.multinomial(mask_prob.float(), num_samples=1).flatten()
        if not self.t5_style:
            masking_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        else:
            masking_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        mask_prob[torch.arange(len(mask_prob)), seq_len - 1] = 0
        ret["input_ids"][torch.arange(len(mask_prob)), mask_seq_token] = masking_id
        return ret, mask_seq_token, org_ids

    def mutate(self, inputs, tag=None):
        ret_list = []
        for k in range(self.n_multiply):
            masked_inputs, masking_indices, org_ids = self._do_masking_replace(inputs)
            ret_list += self._get_topk_replacements(masked_inputs, masking_indices, org_ids=org_ids)
        return ret_list, list(range(len(inputs)))*(self.n_mutations*self.n_multiply)


class WordInsertionMutation(MaskedLMMutationBase):

    def _do_masking_extend(self, sequences):
        """ Mask one random token in each sequence (internal helper function)
            Masks one token in each input sequence randomly.
            Return the masked input (dict), the token_idx of the masked token for each input, and the original input_ids
        """
        ret = self.tokenizer(sequences, padding=True, return_tensors="pt", truncation=True)
        org_ids = ret["input_ids"].clone()
        seq_len = torch.sum(ret["attention_mask"], dim=1)

        mask_prob = ret["attention_mask"].clone()
        # mask_prob[torch.arange(len(mask_prob)), seq_len-1] = 0
        mask_prob[:, 0] = 0

        mask_seq_token = torch.multinomial(mask_prob.float(), num_samples=1).flatten()
        ## Extend attention mask and input_ids by one column
        ret["attention_mask"] = torch.cat(
            (ret["attention_mask"], torch.zeros(len(ret["input_ids"]), 1, dtype=torch.long)), dim=1)
        ret["attention_mask"][torch.arange(len(mask_prob)), seq_len] = 1
        inputs_new = torch.cat((ret["input_ids"], torch.zeros(len(ret["input_ids"]), 1, dtype=torch.long)),
                               dim=1).clone()
        for i in range(len(sequences)):
            inputs_new[i, mask_seq_token[i] + 1:] = ret["input_ids"][i, mask_seq_token[i]:]

        if not self.t5_style:
            masking_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        else:
            masking_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        # shift attention
        inputs_new[torch.arange(len(mask_prob)), mask_seq_token] = masking_id
        ret["input_ids"] = inputs_new
        return ret, mask_seq_token, org_ids

    def mutate(self, inputs, tag=None):
        ret_list = []
        for k in range(self.n_multiply):
            masked_inputs, masking_indices, org_ids = self._do_masking_extend(inputs)
            ret_list += self._get_topk_replacements(masked_inputs, masking_indices, org_ids=org_ids)
        return ret_list, list(range(len(inputs)))*(self.n_mutations*self.n_multiply)

class BackTranslationMutation(ProbCorrectMutation):
    def __init__(self, n_mutations=3, num_beams=2, device="cuda", entail_model="tasksource",
                 miss_prob=0.85, verbose=True):
        super().__init__(device, n_mutations, entail_model, miss_prob)
        self.verbose = verbose
        if self.verbose:
            print("Starting to load English to German Translation Model.\n")
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(
            name_en_de
        ).to(self.device)
        if self.verbose:
            print("Completed loading English to German Translation Model.\n")
            print("Starting to load German to English Translation Model:")
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(
            name_de_en
        ).to(self.device)
        self.num_beams = max(num_beams, self.n_mutations)
        if self.verbose:
            print("Completed loading German to English Translation Model.\n")

    def _back_translate(self, en: str):
        #try:
        de = self._en2de(en)
        en_new = self._de2en(de)
        #except Exception:
        #    print("Returning Default due to Run Time Exception")
        #    en_new = en
        return en_new

    def _en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        outputs = self.model_en_de.generate(input_ids.to(self.device))
        decoded = self.tokenizer_en_de.decode(
            outputs[0], skip_special_tokens=True
        )
        if self.verbose:
            print(decoded)  # Maschinelles Lernen ist großartig, oder?
        return decoded

    def _de2en(self, input):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
        outputs = self.model_de_en.generate(
            input_ids.to(self.device),
            num_return_sequences=self.n_mutations,
            num_beams=self.num_beams,
        )
        predicted_outputs = []
        for output in outputs:
            decoded = self.tokenizer_de_en.decode(
                output, skip_special_tokens=True
            )
            # TODO: this should be able to return multiple sequences
            predicted_outputs.append(decoded)

        if self.verbose:
            print(predicted_outputs)  # Machine learning is great, isn't it?
        return predicted_outputs

    def mutate(self, inputs, tag=None):
        ret_list = []
        backref_list = []
        for backref, sample in enumerate(inputs):
            perturbs = self._back_translate(sample)
            backref_list.extend([backref]*len(perturbs))
            ret_list.extend(perturbs)
        return ret_list, np.array(backref_list)


