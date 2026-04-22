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
    get_instruction_funcname,
    get_instruction_funcparam_number,
    get_instruction_funcparam_string,
    get_instruction_funcparam_name,
    get_instruction_funcparam_nouns,
)


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
            if param == "s" or "string" in param:
                opts: list[str] = extract_strings(prompt)
                return (
                    get_instruction_funcparam_string(
                        prompt, func_def, param, opts
                    ), opts
                )
            opts: list[str] = extract_nouns(prompt)
            return (
                get_instruction_funcparam_nouns(
                    prompt, func_def, param, opts
                ), opts
            )
        opts: list[str] = extract_numbers(prompt)
        return (
            get_instruction_funcparam_number(
                prompt, func_def, param, opts
            ),
            opts
        )

    def decode_options(self, options: list[str],
                       instruction: str) -> str:
        llm: Small_LLM_Model = self.llm

        def score(option: str) -> float:
            ids: list[int] = llm.encode(instruction + option).tolist()[0]
            logits: list[float] = llm.get_logits_from_input_ids(ids)
            return max(logits)

        return max(options, key=score)

    def decode_func_params(self, prompt: str,
                           func_def: CallMeFunction) -> dict[str, str]:
        already_got: set[str] = set()
        result: dict[str, str] = {}

        print(func_def.name, ":")
        for param in func_def.parameters.keys():
            instruction, options = self.get_instruction_funcparam(
                prompt, func_def, param
            )
            options = [opt for opt in options if opt not in already_got]
            option: str = self.decode_options(
                options, instruction
            )
            already_got.add(option)
            result[param] = option
            print(f"{param} = {option}")
        return result

    def decode_func_name(self, prompt: str, func_names: set[str],
                         func_defs: dict[str, CallMeFunction]) -> str:
        llm: Small_LLM_Model = self.llm
        decoded_func: str = "fn_"
        instruction: str = get_instruction_funcname(prompt, func_defs)
        func_names.add("fn_none")

        print(f"\n\n{prompt}")
        while True:
            ids: list[int] = llm.encode(instruction + decoded_func).tolist()[0]
            logits: list[float] = llm.get_logits_from_input_ids(ids)
            valid_tokens: list[Token] = vocab_filter_funcsname_prefix(
                self.vocab, func_names, decoded_func
            )
            if not valid_tokens:
                return "fn_none"
            best_token: Token = max(
                valid_tokens, key=lambda token: logits[token.id]
            )
            decoded_func += best_token.str
            if decoded_func in func_names:
                return decoded_func
