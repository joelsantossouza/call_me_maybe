from llm_sdk import Small_LLM_Model
from src.structures import Token
from src.callme_files_loader import CallMeFunction
from src.helpers import (
    llm_vocab_load,
    vocab_filter_funcsname_prefix,
    extract_numbers,
    get_instruction_funcname,
    get_instruction_funcparam_number
)


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    def decode_func_param_number(self, prompt: str,
                                 func_def: CallMeFunction,
                                 param: str,
                                 nbr_options: list[str]) -> str:
        llm: Small_LLM_Model = self.llm

        def score(nbr: str) -> float:
            instruction: str = get_instruction_funcparam_number(
                prompt, func_def, param, nbr_options
            )
            ids: list[int] = llm.encode(instruction + nbr).tolist()[0]
            logits: list[float] = llm.get_logits_from_input_ids(ids)
            return max(logits)

        return max(nbr_options, key=score)

    def decode_func_params(self, prompt: str,
                           func_def: CallMeFunction) -> dict[str, str]:
        nbr_options: list[str] = extract_numbers(prompt)
        print(func_def.name, ":")
        for param in func_def.parameters.keys():
            nbr: str = self.decode_func_param_number(
                prompt, func_def, param, nbr_options
            )
            nbr_options.remove(nbr)
            print(f"{param} = {nbr}")

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
