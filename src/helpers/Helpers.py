import json
import re
from llm_sdk import Small_LLM_Model
from src.callme_files_loader import CallMeFunction
from typing import Dict, List, Any


def llm_vocab_load(llm: Small_LLM_Model) -> dict[str, int]:
    """Get the LLM vocab json path and convert into a dictionary"""
    vocab_path: str = llm.get_path_to_vocab_file()

    with open(vocab_path, "r") as vocab:
        return json.load(vocab)


STOP_WORDS = {
    "all", "with", "in", "on", "at", "the", "a", "an",
    "and", "or", "of", "to", "for", "by", "is", "are", "be",
    "substitute", "every", "each", "replace", "using", "use",
    "where", "that", "this", "it", "its"
}


def build_prefix_trie(fn_token_sequences: Dict[str, List[int]]
                      ) -> Dict[str, Any]:
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
    Build the instruction text that tells the LLM to output
    only a function name.
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
    """Extract integer values only, excluding floats and float parts."""
    return re.findall(r"(?<!\d\.)(?<!\.)(-?\b\d+\b)(?!\.\d)", prompt)


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
        "- Descriptive words like 'word', 'digit', 'character' are labels "
        "that identify WHAT follows, not values themselves. The value is what "
        "comes immediately after the label, even if it is quoted.\n"
        "- Example: 'the word cat' → label='word', value='cat'\n"
        "- Example: 'the digit 9' → label='digit', value='9'\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{param} = "
    )
