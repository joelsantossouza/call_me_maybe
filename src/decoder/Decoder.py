from typing import Dict
from llm_sdk import Small_LLM_Model
from src.structures import Token
from src.callme_files_loader import CallMeFunction
from src.helpers import (
    llm_vocab_load,
    vocab_filter_funcsname_prefix,
    extract_numbers,
    extract_strings,
    extract_names,
    extract_nouns,
    extract_keywords,
    get_instruction_funcname,
    get_instruction_funcparam_number,
    get_instruction_funcparam_string,
    get_instruction_funcparam_name,
    get_instruction_funcparam_replacement,
    get_instruction_funcparam_regex,
)

from typing import Dict, List, Any

MAX_FUNC_NAME_TOKENS = 20


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


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    @staticmethod
    def get_instruction_funcparam(prompt: str,
                                  func_def: CallMeFunction,
                                  param: str) -> tuple:
        func_param_type: str = func_def.parameters[param].type
        if func_param_type == "string":
            if param == "name":
                opts: list[str] = extract_names(prompt)
                return (
                    get_instruction_funcparam_name(
                        prompt, func_def, param, opts
                    ), opts
                )
            if "regex" in param:
                opts: list[str] = extract_nouns(prompt)
                return (
                    get_instruction_funcparam_regex(
                        prompt, func_def, param, opts
                    ), opts
                )
            if "replace" in param or param in ("database", "encoding"):
                opts: list[str] = extract_nouns(prompt)
                return (
                    get_instruction_funcparam_replacement(
                        prompt, func_def, param, opts
                    ), opts
                )
            opts: list[str] = extract_strings(prompt)
            print(opts)
            return (
                get_instruction_funcparam_string(
                    prompt, func_def, param, opts
                ), opts
            )
        opts: list[str] = extract_numbers(prompt)
        return (
            get_instruction_funcparam_number(
                prompt, func_def, param, opts
            ), opts
        )

    def decode_options(self, options: list[str],
                       instruction: str) -> str:
        llm: Small_LLM_Model = self.llm

        if not options:
            return "none"

        if len(options) == 1:
            return options[0]

        def score(option: str) -> float:
            ids: list[int] = llm.encode(instruction + option).tolist()[0]
            logits: list[float] = llm.get_logits_from_input_ids(ids)
            return max(logits)

        return max(options, key=score)

    def decode_func_params(self, prompt: str,
                           func_def: CallMeFunction) -> dict[str, str]:
        already_got: set[str] = set()
        result: dict[str, str] = {}

        for param in reversed(list(func_def.parameters.keys())):
            instruction, options = self.get_instruction_funcparam(
                prompt, func_def, param
            )
            options = [opt for opt in options if opt not in already_got]
            option: str = self.decode_options(
                options, instruction
            )
            already_got.add(option)
            result[param] = option
        return dict(reversed(list(result.items())))

    def decode_func_name(
        self,
        prompt: str,
        func_names: set[str],
        func_defs: dict[str, CallMeFunction]
    ) -> str:
        """
        Select a function name using constrained decoding.
        No heuristics. No keyword scoring.
        """

        # 1. Build vocabulary mapping
        vocab = self.vocab

        # 2. Encode all function names into token sequences
        fn_token_sequences = {
            name: self.llm.encode(name).tolist()[0]
            for name in func_names
        }

        # 3. Build prefix trie for allowed token sequences
        trie = build_prefix_trie(fn_token_sequences)

        # 4. Build the initial prompt
        instruction = build_instruction_for_func_name(prompt, func_defs)
        input_ids = self.llm.encode(instruction).tolist()[0]

        generated: list[int] = []

        while True:
            # 5. Get logits for next token
            logits = self.llm.get_logits_from_input_ids(input_ids + generated)

            # 6. Filter logits using constrained decoding
            for token_id in range(len(logits)):
                candidate_sequence = generated + [token_id]

                if not is_valid_prefix(candidate_sequence, trie):
                    logits[token_id] = float('-inf')

            # 7. Select next token
            next_token = max(enumerate(logits), key=lambda x: x[1])[0]
            generated.append(next_token)

            # 8. Check if we reached a full function name
            decoded = self.llm.decode(generated)
            if decoded in func_names:
                return decoded

            # 9. Safety: prevent infinite loops
            if len(generated) > MAX_FUNC_NAME_TOKENS:
                return "fn_none"
