import math
from typing import Dict
from llm_sdk import Small_LLM_Model
from src.callme_files_loader import CallMeFunction
from src.helpers import (
    llm_vocab_load,
    vocab_filter_funcsname_prefix,
    get_instruction_funcname,
    get_instruction_funcparam_number,
    get_instruction_funcparam_string,
    get_instruction_funcparam_name,
    get_instruction_funcparam_replacement,
    get_instruction_funcparam_regex,
)

import re
from typing import Dict, List, Any

MAX_TOKENS = 20
STOP_WORDS = {
    "replace", "all", "with", "in", "on", "at", "the", "a", "an",
    "and", "or", "of", "to", "for", "by", "is", "are", "be"
}


def build_prefix_trie(fn_token_sequences: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Build a prefix trie from token sequences.
    Each path in the trie corresponds to a valid function name.
    """
    root: Dict[str, Any] = {}

    for seq in fn_token_sequences.values():
        node = root
        for token_id in seq:
            if token_id not in node:
                node[token_id] = {}
            node = node[token_id]
        node["__END__"] = True

    return root


def build_instruction_for_func_name(
    prompt: str,
    func_defs: Dict[str, Any]
) -> str:
    """
    Build the instruction text that tells the LLM to output only a function name.
    """
    fn_list = ", ".join(func_defs.keys())
    return (
        "You must select exactly one function name from the following list: "
        f"{fn_list}. "
        "Output ONLY the function name, with no punctuation, no quotes, "
        "no explanation. "
        f"User request: {prompt}\nFunction name: "
    )


def is_valid_prefix(candidate: List[int], trie: Dict[str, Any]) -> bool:
    """
    Check whether the candidate token sequence is a valid prefix
    of any function name in the trie.
    """
    node = trie
    for token_id in candidate:
        if token_id not in node:
            return False
        node = node[token_id]
    return True


def extract_numbers(prompt: str) -> list[str]:
    """Extract int and float values."""
    return re.findall(r"-?\d+(?:\.\d+)?", prompt)


def extract_ints(prompt: str) -> list[str]:
    """Extract integer values."""
    return re.findall(r"-?\d+", prompt)


def extract_strings(prompt: str) -> list[str]:
    """
    Extract:
      - quoted substrings as single units
      - unquoted nouns (alphabetic words)
    Without splitting quoted strings into words.
    """
    results = []

    # 1. Extract quoted substrings
    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', prompt)
    quoted = [q[0] or q[1] for q in quoted]

    results.extend(quoted)

    # 2. Remove quoted substrings from the prompt
    cleaned = re.sub(r'"[^"]*"|\'[^\']*\'', ' ', prompt)

    # 3. Extract nouns (alphabetic words)
    nouns = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned)

    results.extend(nouns)

    filtered = [
        s for s in results
        if s.lower() not in STOP_WORDS
    ]
    return filtered


def build_instruction_for_func_params(
    prompt: str,
    param: str,
    options: list[str],
    func: CallMeFunction
) -> str:

    opts_str = " | ".join(options)

    return (
        "<|im_start|>system\n"
        "You select EXACTLY ONE value for a function parameter.\n\n"

        "You MUST understand the transformation described by the user.\n"
        "Think in terms of:\n"
        "- original text\n"
        "- what is searched\n"
        "- what replaces it\n\n"

        "STRICT RULES:\n"
        "- source_string = the FULL original text\n"
        "- regex = what is searched inside the text (or a category like 'numbers', 'vowels')\n"
        "- replacement = what is inserted instead\n\n"

        "HARD CONSTRAINTS (CRITICAL):\n"
        "- source_string MUST be the longest text span\n"
        "- regex MUST NOT be the full sentence\n"
        "- replacement MUST NOT be part of the original sentence\n"
        "- regex and replacement MUST NOT be swapped\n\n"

        "If a candidate violates these constraints, DO NOT select it.\n\n"

        "Output ONLY the selected value.\n"
        "<|im_end|>\n"

        f"<|im_start|>user\n"
        f"Function: {func.name}\n"
        f"Description: {func.description}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Parameter: {param}\n"
        f"Options: {opts_str}\n"
        f"<|im_end|>\n"

        f"<|im_start|>assistant\n"
    )


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    def extract_params_options(self, func: CallMeFunction,
                               param: str, prompt: str) -> list[str]:
        """
        Given a parameter it extract the possible values by its
        type(number | integer | string)
        """
        ptype = func.parameters[param].type

        if ptype == "number":
            options = extract_numbers(prompt)

        elif ptype == "integer":
            options = extract_ints(prompt)

        else:  # string
            options = extract_strings(prompt)
        return options

    def constrained_decode_from_options(
        self,
        instruction: str,
        options: list[str]
    ) -> str:
        """
        Generic constrained decoder that selects exactly one string from a list
        of allowed options using token-by-token constrained decoding.
        """
        if not options:
            return "none"

        if len(options) == 1:
            return options[0]

        llm = self.llm

        # Encode all options into token sequences
        option_token_sequences = {
            opt: llm.encode(opt).tolist()[0] for opt in options
        }

        # Build prefix trie
        trie = build_prefix_trie(option_token_sequences)

        # Encode instruction
        input_ids = llm.encode(instruction).tolist()[0]

        generated: list[int] = []

        while True:
            # Get logits
            logits = llm.get_logits_from_input_ids(input_ids + generated)

            # Mask invalid tokens
            for token_id in range(len(logits)):
                candidate = generated + [token_id]
                if not is_valid_prefix(candidate, trie):
                    logits[token_id] = float('-inf')

            # Select next token
            next_token = max(enumerate(logits), key=lambda x: x[1])[0]
            generated.append(next_token)

            # Decode partial output
            decoded = llm.decode(generated)

            # Check if we reached a full option
            if decoded in options:
                return decoded

            # Safety
            if len(generated) > MAX_TOKENS:
                return options[0]

    def decode_func_params(self, prompt, func_def):
        """
        Decode all parameters of a function using constrained decoding.
        For each parameter:
          - extract candidate options
          - build an instruction
          - use constrained decoding to choose exactly one option
        """
        result = {}
        used = set()   # track already chosen values

        for param in func_def.parameters.keys():
            options = self.extract_params_options(func_def, param, prompt)

            # Remove already-used values
            options = [opt for opt in options if opt not in used]

            instruction = build_instruction_for_func_params(
                prompt, param, options, func_def
            )

            chosen = self.constrained_decode_from_options(instruction, options)
            result[param] = chosen

            used.add(chosen)   # mark as used

        return result

    def decode_func_name(
        self,
        prompt: str,
        func_names: set[str],
        func_defs: dict[str, CallMeFunction]
    ) -> str:
        instruction = build_instruction_for_func_name(prompt, func_defs)
        options = list(func_names)
        return self.constrained_decode_from_options(instruction, options)
