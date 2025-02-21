from typing import List, Union
from litellm import batch_completion
import litellm
import os
from tqdm import tqdm
import time
import numpy as np


def generate_prompt_request(system_prompt: str, documents: List[str]) -> List[str]:
    """
    Format the LLM prompts for this batch.
    """

    generate_prompt = lambda prefix, doc: f"""
        {prefix}  The document is: 
         <document> {doc} </document>
         Assistant: """

    prompts = [
        generate_prompt(system_prompt, doc) for doc in documents
    ]

    return prompts


def extract_tag(llm_response: str, tag_name: str) -> str:
    pos_start = llm_response.find(f'<{tag_name}>')
    pos_end = llm_response.find(f'</{tag_name}>')
    return llm_response[pos_start+len(tag_name)+2:pos_end]


def get_bedrock_batch_response(
        prompts: List[str],
        model: str,
        temperature: float = 0.0,
        n_choices: int =1,
        max_new_tokens: int =4096,
        wait_seconds: int =10) -> List[str]:
    """
    Get batch generation results with given prompts.

    Parameters
    ----------
    prompts : List[str]
        List of prompts for generation.
    temperature : float, optional
        The generation temperature, use greedy decoding when setting
        temperature=0, defaults to 0.
    model : str, optional
        The model for generation, defaults to 'gpt-4'.
    n_choices : int, optional
        How many samples to return for each prompt input, defaults to 1.
    max_new_tokens : int, optional
        Maximum number of newly generated tokens, defaults to 500.

    Returns
    -------
    response_list : List[str]
        List of generated text.
    """

    if not prompts or len(prompts) == 0:
        raise ValueError("Invalid input.")

    message_list = []
    for prompt in prompts:
        if len(prompt) == 0:
            raise ValueError("Invalid prompt.")
        if isinstance(prompt, str):
            messages = [{
                'role': 'user',
                'content': prompt
            }]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Invalid prompt type.")
        message_list.append(messages)
    litellm.suppress_debug_info = True
    # litellm.set_verbose = True
    while True:
        responses = None
        try:
            responses = batch_completion(
                model=model,
                messages=message_list,
                temperature=temperature,
                n=n_choices,
                max_tokens=max_new_tokens
            )
            if n_choices == 1:
                response_list = [r.choices[0].message.content for r in responses]
            else:
                response_list = [[res.message.content for res in r.choices] for r in responses]
            for r in response_list:
                if not r or len(r) == 0:
                    raise ValueError(f'{model} API returns None or empty string')
            return response_list
        except Exception as e:
            if responses is not None:
                print(responses)
            print(e)
            print(f"[sleep {wait_seconds} seconds]")
            time.sleep(wait_seconds)
            wait_seconds = min(60, 2 * wait_seconds)

    raise "Something went wrong."

