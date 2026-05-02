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
    "all", "with", "in", "on", "at", "the", "a", "an",
    "and", "or", "of", "to", "for", "by", "is", "are", "be",
    "substitute", "every", "each", "replace", "using", "use",
    "where", "that", "this", "it", "its"
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
    """Extract int and float values, always returned as float strings."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return [str(float(m)) for m in matches]


def extract_ints(prompt: str) -> list[str]:
    """Extract integer values only, excluding floats."""
    return re.findall(r"-?\b\d+\b(?!\.\d)", prompt)


def extract_strings(prompt: str) -> list[str]:
    """
    Extract:
      - quoted substrings as single units
      - colon-introduced substrings: text after ': ' (colon followed by space)
        until a terminator (. ! ?) or end of string
      - unquoted tokens: start with anything except a digit or whitespace,
        end at whitespace
    Without splitting quoted strings into words.
    """
    results = []

    colon_values = re.findall(r'(?<=\S):\s+([^.!?]+?)(?:[.!?]|$)', prompt)
    colon_values = [v.strip() for v in colon_values if v.strip()]
    results.extend(colon_values)

    cleaned = re.sub(r'(?<=\S):\s+[^.!?]+?(?:[.!?]|$)', ' ', prompt)

    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', cleaned)
    quoted = [q[0] or q[1] for q in quoted]
    results.extend(quoted)

    cleaned = re.sub(r'"[^"]*"|\'[^\']*\'', ' ', cleaned)

    tokens = re.findall(r'[^\d\s][^\s]*', cleaned)
    results.extend(tokens)

    filtered = [
        s for s in results
        if s.lower() not in STOP_WORDS
    ]
    return filtered


def build_instruction_for_func_params(
    prompt: str,
    param: str,
    options: list[str],
    func_def: CallMeFunction
) -> str:
    ptype = func_def.parameters[param].type
    opts = " | ".join(options)

    all_params_context = "\n".join(
        f"- {p} ({func_def.parameters[p].type})"
        for p in func_def.parameters
    )

    return (
        "<|im_start|>system\n"
        f"You are assigning a value to one parameter of the function `{
            func_def.name}`.\n\n"
        f"Function purpose: {func_def.description}\n\n"
        "The function has the following parameters:\n"
        f"{all_params_context}\n\n"
        f"You must assign a value to `{param}` ({ptype}).\n"
        f"Choose exactly one of: {opts}\n\n"
        "Rules:\n"
        "- Output ONLY the chosen value, nothing else.\n"
        "- Choose ONLY from the provided options.\n"
        "- Do not invent, combine, or modify values.\n"
        "- Descriptive words like 'word', 'digit', 'character' are labels that "
        "identify WHAT follows, not values themselves. The value is what comes "
        "immediately after the label, even if it is quoted.\n"
        "- Example: 'the word cat' → label='word', value='cat'\n"
        "- Example: 'the digit 9' → label='digit', value='9'\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{param} = "
    )


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    def extract_params_options(
        self,
        func: CallMeFunction,
        param: str,
        prompt: str
    ) -> list[str]:
        """
        Given a parameter it extract the possible values by its
        type(number | integer | string),
        """
        ptype = func.parameters[param].type

        if ptype == "number":
            options = extract_numbers(prompt)
        elif ptype == "integer":
            options = extract_ints(prompt)
        else:
            options = extract_strings(prompt)

        print(options)
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
        Parameters are *processed* in descending name-length order,
        but the final dictionary preserves the original order.
        """
        result = {}
        used = set()

        # 1. Get original order
        original_params = list(func_def.parameters.keys())

        # 2. Sort by descending length (longest name first)
        sorted_params = sorted(original_params, key=len, reverse=True)

        # 3. Temporary storage for decoded values
        temp_values = {}

        # 4. Decode in sorted order
        for param in sorted_params:
            options = self.extract_params_options(func_def, param, prompt)

            # Remove already-used values
            options = [opt for opt in options if opt not in used]

            instruction = build_instruction_for_func_params(
                prompt, param, options, func_def
            )

            chosen = self.constrained_decode_from_options(instruction, options)
            temp_values[param] = chosen
            used.add(chosen)

        # 5. Reconstruct result in original order
        for param in original_params:
            result[param] = temp_values[param]

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
