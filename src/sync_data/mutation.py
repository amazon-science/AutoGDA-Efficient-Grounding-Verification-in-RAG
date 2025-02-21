## Implement different sample mutation strategies here
## DEPRICATED -> See pc_mutations for current mutations.

import abc
from abc import ABC
import typing as tp
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from src.sync_data.population import Population
import torch
from tqdm import tqdm
import os
from src.utils.bedrock_utils import get_bedrock_batch_response
from multiprocessing import Pool
from random import sample

def repeat_sorted(n_max, n_repeats):
    """ return list [0,0,0, 1,1,1, ... n_max, n_max, n_max], where
    there are n_repeats of each number (3 in this example)"""
    return sum([[i] * n_repeats for i in range(n_max)], [])

class Mutation(ABC):
    def __init__(self, device, n_mutations):
        self.n_mutations = n_mutations
        self.device = device

    def mutate(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """ Our mutation interface. Mutates the samples in inputs and creates self.n_mutations offspring.
            :param: inputs: A list of input samples.
            :return: A mutated list of input samples. The length of output list is len(inputs)*self.n_mutations.
                     Additionally return a list of the indices of the original samples that each mutation was created from.
        """
        raise NotImplementedError()


    def mutate_all_tags(self, input_population: Population):
        for tag in tqdm(input_population.tags, total=len(input_population.tags)):
            input_population[tag] = self.mutate(input_population[tag].tolist(), tag)
        return input_population

    def __call__(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """ Our mutation interface. Mutates the samples in inputs. See Mutation.mutate for details. """
        return self.mutate(inputs, tag)



class MutationList():
    def __init__(self, *mutations):
        self.mymutations = list(mutations)

    def mutate(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        output_mut, output_ref = [], []
        for m in self.mymutations:
            mutations, refs = m(inputs, tag)
            output_mut += mutations
            output_ref += refs
        return output_mut, output_ref

    def mutate_all_tags(self, input_population: Population):
        p = Population()
        for m in self.mymutations:
            children = m.mutate_all_tags(input_population)
            p = p + children
        return p

    def extend(self, list_mutations: tp.Union[tp.List[Mutation], "MutationList"]):
        if isinstance(list_mutations, MutationList):
            self.mymutations += list_mutations
        else:
            self.mymutations += list_mutations

    def __call__(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """ Our mutation interface. Mutates the samples in inputs. See Mutation.mutate for details. """
        return self.mutate(inputs, tag)

    def __len__(self):
        return len(self.mymutations)

class RephraseMutation(Mutation):
    """ Use a paraphrasing model to do the mutation step. """
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
                 ):
        super(RephraseMutation, self).__init__(device=device, n_mutations=n_mutations)
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

class WordDeletionMutation(Mutation):
    """ Deletes random tokens from an input. """
    def __init__(self, device='cuda', n_mutations=3, tokenizer_str="FacebookAI/roberta-large"):
        super().__init__(device=device, n_mutations=n_mutations)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.t5_style = ("t5" in tokenizer_str)

    def mutate(self, inputs, tag=None):
        """ Cut a piece of text by deleting one token. """
        ret = self.tokenizer(inputs, padding=True, return_tensors="pt")
        repeat_ids = ret["input_ids"].unsqueeze(0).expand(self.n_mutations, *ret["input_ids"].shape)
        seq_len = torch.sum(ret["attention_mask"], dim=1)

        repeat_mask = ret["attention_mask"].unsqueeze(0).expand(self.n_mutations, *ret["attention_mask"].shape).clone()
        if not self.t5_style:  # No deletion of last token.
            repeat_mask[:, :, seq_len - 1] = 0
        mask_prob = repeat_mask.reshape(-1, repeat_mask.size(2))
        repeat_ids = repeat_ids.reshape(-1, repeat_ids.size(2))
        repeat_ids_new = repeat_ids.clone()
        # Sample which token to mask
        mask_seq_token = torch.multinomial(mask_prob.float(), num_samples=1).flatten()
        for i in range(len(repeat_ids)):
            repeat_ids_new[i, mask_seq_token[i]:-1] = repeat_ids[i, mask_seq_token[i] + 1:]
        return self.tokenizer.batch_decode(repeat_ids_new[:, :-1], skip_special_tokens=True), list(range(len(inputs)))*self.n_mutations   #Decode all but last token

class MaskedLMMutationBase(Mutation):
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
                 tokenizer_str="FacebookAI/roberta-large", max_length=512, consider_top_k=200, use_vocab=None,
                 n_multiply = 1):
        """
            :param: max_length: maximum length of the model used.
            :param: consider_top_k: before sampling the top k most likely tokens of the entire vocab of use_vocab are
                selected. Then we sample the required n_mutations form this pool randomly.
            :param: use_vocab: vocabulary considered for replacement. If None, the entire vocabulary will be used.
        """
        super().__init__(device=device, n_mutations=n_mutations)
        self.device = device
        # t5_style indicates whether a MaskedLM or Seq2Seq Architecture is used.
        self.t5_style = ("t5" in model_str)

        if not self.t5_style: #Masked LM
            self.model = AutoModelForMaskedLM.from_pretrained(model_str).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        else: #Seq2Seq LM
            self.tokenizer = AutoTokenizer.from_pretrained(model_str)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_str).to(device)
        self.max_length = max_length
        self.device = device
        self.temperature = 1.0
        self.consider_top_k = consider_top_k
        self.top_k = consider_top_k  # number of topk tokens to consider
        self.vocab = use_vocab
        self.batch_size=64
        self.n_multiply = n_multiply

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

    def _get_topk_replacements(self, ret_masking, replace_ids, use_vocab=None):
        """
            Return the extended inputs, each with k replacements for the originally masked tokens.
            ret_masking: Masked inputs
            replace_ids: The index in each input that was masked.
        """
        if self.t5_style:
            my_masked_lm_probs = self._forward_t5_replacements(ret_masking["input_ids"]).cpu()
        else:
            my_masked_lm_probs = self.model(input_ids=ret_masking["input_ids"].to(self.device),
                                            attention_mask=ret_masking["attention_mask"].to(self.device)).logits.cpu()
            my_masked_lm_probs = my_masked_lm_probs[torch.arange(len(replace_ids)), replace_ids] # only select relevant probabilities

        # my_masked_lm_probs is of shape Batch x Vocab and now contains the logits for each token.
        if use_vocab is not None:
            my_masked_lm_probs = my_masked_lm_probs[:,use_vocab]

        # Find topk indices
        _, indices = torch.topk(my_masked_lm_probs,
                               min(self.top_k, len(use_vocab)) if self.vocab is not None else self.top_k, dim=-1)

        ### Assemble output sequences (sampling randomly from the topk)
        outputs = ret_masking["input_ids"].expand(self.n_mutations, *ret_masking["input_ids"].shape).clone()
        # print(indices[:, :, torch.randperm(indices.shape[2])[:n_outputs]].shape)
        for i in range(len(ret_masking["input_ids"])):
            outputs[:, i, replace_ids[i]] = indices[i, torch.randperm(indices.shape[1])[:self.n_mutations]]
            if use_vocab is not None:  ## indices correspond to the ones in the vocab
                outputs[:, i, replace_ids[i]] = use_vocab[outputs[:, i, replace_ids[i]]]
        return self.tokenizer.batch_decode(outputs.reshape(-1, ret_masking["input_ids"].shape[-1]), skip_special_tokens=True)


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
            if self.vocab is not None:
                ret_list += self._get_topk_replacements(masked_inputs, masking_indices, use_vocab=self.vocab[tag])
            else:
                ret_list += self._get_topk_replacements(masked_inputs, masking_indices)
        return ret_list, list(range(len(inputs)))*(self.n_mutations*self.n_multiply)

class WordExtensionMutation(MaskedLMMutationBase):

    def _do_masking_extend(self, sequences):
        """ Mask one random token in each sequence (internal helper function)
            Masks one token in each input sequence randomly.
            Return the masked input (dict), the token_idx of the masked token for each input, and the original input_ids
        """
        ret = self.tokenizer(sequences, padding=True, return_tensors="pt", truncation=True)
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
        return ret, mask_seq_token

    def mutate(self, inputs, tag=None):
        ret_list = []
        for k in range(self.n_multiply):
            masked_inputs, masking_indices, = self._do_masking_extend(inputs)
            if self.vocab is not None:
                ret_list += self._get_topk_replacements(masked_inputs, masking_indices, use_vocab=self.vocab[tag])
            else:
                ret_list += self._get_topk_replacements(masked_inputs, masking_indices)
        return ret_list, list(range(len(inputs)))*(self.n_mutations*self.n_multiply)

class GuidedReplacementMutation(Mutation):
    """ Use a replacement model based on synoyms similar to Alzantot et al., 2018"""
    def __init__(self):
        raise NotImplementedError("GuidedReplacementMutation is not yet implemented.")


def extract_tag(llm_response: str, tag_name: str) -> str:
    pos_start = llm_response.find(f'<{tag_name}>')
    pos_end = llm_response.find(f'</{tag_name}>')
    if pos_start  == -1 or pos_end == -1:
        print("Invalid response:", llm_response)
        return None

    return llm_response[pos_start+len(tag_name)+2:pos_end]


class APIModelMutation(Mutation):
    """ Call an API model with a prompt to create mutations.
        Note that this is an abstract class. You have to specify a prompt to implement this class.
        The output will generally be expected between tags <answer #>Answer<Answer 1>.
        The final prompt will be created by formating the prompt with (n_mutations, n_mutation-1) and appending the doc.
    """

    def __init__(self, device=None, n_mutations=3, model_name="claude3-haiku", batch_size=10,
                 temperature = 0.0, n_threads=6):
        os.environ["AWS_REGION_NAME"] = 'us-east-1'
        if model_name == 'claude3-sonnet':
            self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        elif model_name == 'claude3-haiku':
            self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        else:
            raise ValueError(f"Unrecognized model name: {model_name}")
        
        super().__init__(device, n_mutations)
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_threads=n_threads

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
            generate_prompt(self.get_base_prompt().format(self.n_mutations, self.n_mutations-1), doc) for doc in documents
        ]
        #print(prompts[0])
        return prompts

    @abc.abstractmethod
    def get_base_prompt(self):
        raise NotImplementedError("Not implemented in this abstract class.")

    def mutate(self, inputs: tp.List[str], tag=None) -> tp.Tuple[tp.List[str], tp.List[int]]:
        ret = mutate_thread([self, inputs, tag])
        return ret[1], ret[2]

    def mutate_all_tags(self, input_population: Population):
        pop = Population()
        print(f"Using {self.n_threads} threads.")
        tasklist = []
        for tag in tqdm(input_population.tags, total=len(input_population.tags)):
            tasklist.append((self, input_population[tag].tolist(), tag))
        with Pool(self.n_threads) as p:
            res_array = p.map(mutate_thread, tasklist)
        for k, v, r in res_array:
            pop[k] = v, r
        return pop

def mutate_thread(arglist):
    """
        Performs the LLM call and returns tuple (tag, outputs, references)
    """
    mutation_model, inputs, tag = arglist
    inputs = np.array(inputs)
    output_list = []
    for doc_batch in np.array_split(inputs, (len(inputs) // mutation_model.batch_size) + 1):
        prompts = mutation_model._generate_prompt_request(doc_batch, tag)
        ## Resubmit query until valid response was received.
        invalid = True
        while invalid:
            valid_answers = []
            invalid = False
            llm_responses = get_bedrock_batch_response(prompts, model=mutation_model.model_id, temperature=mutation_model.temperature)
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
    return tag, output_list, repeat_sorted(len(inputs), mutation_model.n_multiply*mutation_model.n_mutations)

class LLMFillInTheGapsMutation(APIModelMutation):
    """ Mask parts of the sentence and let them be replaced by an LLM. """
    def __init__(self, device=None, n_mutations=3, model_name="claude3-haiku",
                 batch_size=10, temperature=1.0, mask_output_perc=0.2, connected=True, n_multiply=1,
                 n_threads=6, preserve_meaning=False):
        super().__init__(device=device, n_mutations=n_mutations, model_name=model_name,
                         batch_size=batch_size, temperature=temperature, n_threads=n_threads)
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
                            {prefix} You will now see the original document, but you will have to generate different versions that have the similar meaning by filling the gaps.
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
                    res = np.random.randint(num_toks - num_mask)
                    parts[res:res + num_mask] = ["_"] * num_mask
                else:
                    res = np.random.randint(num_toks, size=num_mask)
                    for i in range(num_mask):
                        parts[res[i]] = "_"
                document_gaps = " ".join(parts)
                if self.preserve_meaning:
                    prompts.append(generate_prompt_org(self.get_base_prompt().format(self.n_mutations, self.n_mutations-1),
                                                       documents[i], document_gaps))
                else:
                    prompts.append(generate_prompt(self.get_base_prompt().format(self.n_mutations, self.n_mutations-1),
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


class LLMTargetedFillInTheGaps(APIModelMutation):
    """ Mask parts of the sentence and let them be replaced by an LLM. """

    def __init__(self, word_list=tp.Dict[tp.Any, set], device=None, n_mutations=3, model_name="claude3-haiku",
                 batch_size=10, temperature=1.0, mask_output_perc=0.2, connected=True, n_multiply=1,
                 n_threads=6):
        super().__init__(device=device, n_mutations=n_mutations, model_name=model_name,
                         batch_size=batch_size, temperature=temperature, n_threads=n_threads)
        """ Arguments:
            connected: Whether the masked words should be next to each other or not.

        """
        self.mask_output_perc = mask_output_perc
        self.connected = connected
        self.n_multiply = n_multiply
        self.word_list = word_list

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
        for k in range(self.n_multiply):
            # Split and Mask
            for i in range(len(documents)):
                parts = documents[i].split(" ")
                num_toks = len(parts)
                num_mask = min(int(len(parts) * self.mask_output_perc + 1.0), len(parts))
                if self.connected:
                    res = np.random.randint(num_toks - num_mask)
                    parts[res:res + num_mask] = ["_"] * num_mask
                else:
                    res = np.random.randint(num_toks, size=num_mask)
                    for i in range(num_mask):
                        parts[res[i]] = "_"
                document_gaps = " ".join(parts)
                ## Check for words not in the original doc
                set_parts = list(self.word_list[tag].difference(parts))
                ## Format accordingly
                n_target_words = min(10, len(set_parts))
                set_parts_use = sample(set_parts, n_target_words)
                target_words = ", ".join(['"' + w + '"' for w in set_parts_use])
                #print(target_words)

                prompts.append(
                    generate_prompt(self.get_base_prompt().format(target_words, self.n_mutations, self.n_mutations - 1),
                                    document_gaps))
        return prompts

    def get_base_prompt(self):
        return """ Your task is to fill in the gaps in a document indicated with "_" with additional details. 
            If there is no gaps, please output the input text. The number of "_" indicates the approximate number of words
            that should be filled into each gap. **Do not change the text outside the gaps and do not include gaps in the final output.**
            Additionally, your are provided with a list of words: {}. If possible, use one of these words in your solution.
            You will generate {} semantically different completions of the document. Each completed document is identified with an integer from 0 to {}. 
            The document with the blanks filled must be wrapped in <answer #></answer #> tags, where # is replaced with the id of the filled-in document.
        """

class APIRephrasingMutation(APIModelMutation):
    def get_base_prompt(self):
        return """Human: You will be given a short text document wrapped in <document></document> tags. 
        Your task is to generate {} grammatically consistent rephrased versions of the document. 
        The rephrased versions must be different from each other. They can have diverse linguistic styles (formal, informal, written, spoken ...).
        The length (number of words) should roughly reflect the original input.
        Each rephrased document is identified with an integer from 0 to {}. 
        The rephrased document  must be wrapped in <answer #></answer #> tags, where # is replaced with the id of the rephrased document."""

class APIGrammarCorrection(APIModelMutation):
    def get_base_prompt(self):
        return """Human: You will be given a short text document wrapped in <document></document> tags.
        Your task is to generate {} grammatically consistent versions of the document. Try to change as many words as possible, while making the input a gramatically consistent text. Stay as close to the original as possible.
        The length (number of words) should closely reflect the original input.
        Each corrected document is identified with an integer from 0 to {}. 
        The corrected document must be wrapped in <answer #></answer #> tags, where # is replaced with the id of the corrected document."""

"""
    def _check_nli(self, seeds: tp.List[str], summaries: tp.List[str], n_mutations: int):
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base').to(self.device)
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
        tuple_list = []
        for i in range(len(summaries)):
            tuple_list.append([seeds[i//self.sample_paraphrases], summaries[i]])

        features = tokenizer(tuple_list, padding=True, truncation=True, return_tensors="pt")
        model.eval()
        score_list = []
        bs=self.batch_size
        with torch.no_grad():
            input_ids = features["input_ids"].to(self.device)
            attn_mask = features["attention_mask"].to(self.device)
            for i in range((len(tuple_list) // bs) + 1):
                scores = torch.softmax(model(input_ids[i*bs:(i+1)*bs], attention_mask=attn_mask[i*bs:(i+1)*bs]).logits, dim=-1)[:, 1].cpu()  # column one is for entailment.
                score_list.append(scores)
        scores = torch.cat(score_list, dim=0)
        ## Rerank according to the scores.
        scores = scores.reshape(len(seeds), -1)
        topk, indices = torch.topk(scores, n_mutations, dim=-1)
        filtered_list = []
        for i in range(len(seeds)):
            filtered_list.extend(summaries[self.sample_paraphrases*i + k] for k in indices[i])
        return filtered_list
"""

if __name__ == "__main__":
    ### Some test code
    mymutator = MaskedLMMutation(model_str="FacebookAI/roberta-large", tokenizer_str="FacebookAI/roberta-large")
    #mymutator = APIModelMutation()
    print(mymutator(['New York is a great place to live.',
               'The Eiffel Tower is a monument and popular tourist attraction in Paris, France.'], n_mutations=3))
    #mymutator = RephraseMutation()
    #print(mymutator(['New York is a great place to live.',
    #                 'The Eiffel Tower is a monument and popular tourist attraction in Paris, France.']))