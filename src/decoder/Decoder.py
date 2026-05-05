from llm_sdk import Small_LLM_Model
from src.callme_files_loader import CallMeFunction
from src.helpers import (
    build_prefix_trie,
    build_instruction_for_func_name,
    build_instruction_for_func_params,
    is_valid_prefix,
    extract_ints,
    extract_numbers,
    extract_strings,
)

MAX_TOKENS = 20


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()

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

        clean = set()
        clean.add(func.name.lower())
        for part in func.name.lower().split("_"):
            clean.add(part)
        for p in func.parameters.keys():
            p_lower = p.lower()
            clean.add(p_lower)
            for part in p_lower.split("_"):
                clean.add(part)

        cleaned = [
            opt for opt in options
            if opt.lower() not in clean
        ]

        return cleaned

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

        option_token_sequences = {
            opt: llm.encode(opt).tolist()[0] for opt in options
        }

        trie = build_prefix_trie(option_token_sequences)

        input_ids = llm.encode(instruction).tolist()[0]

        generated: list[int] = []

        while True:
            logits = llm.get_logits_from_input_ids(input_ids + generated)

            for token_id in range(len(logits)):
                candidate = generated + [token_id]
                if not is_valid_prefix(candidate, trie):
                    logits[token_id] = float('-inf')

            next_token = max(enumerate(logits), key=lambda x: x[1])[0]
            generated.append(next_token)

            decoded = llm.decode(generated)

            if decoded in options:
                return decoded

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

        original_params = list(func_def.parameters.keys())

        sorted_params = sorted(original_params, key=len, reverse=True)

        temp_values = {}

        for param in sorted_params:
            options = self.extract_params_options(func_def, param, prompt)

            options = [opt for opt in options if opt not in used]

            instruction = build_instruction_for_func_params(
                prompt, param, options, func_def
            )

            chosen = self.constrained_decode_from_options(instruction, options)
            temp_values[param] = chosen
            used.add(chosen)

        for param in original_params:
            result[param] = temp_values[param]

        return result

    def decode_func_name(
        self,
        prompt: str,
        func_names: set[str],
        func_defs: dict[str, CallMeFunction]
    ) -> str:
        if not prompt.strip():
            return "fn_none"

        instruction = build_instruction_for_func_name(prompt, func_defs)
        options = list(func_names)
        return self.constrained_decode_from_options(instruction, options)
