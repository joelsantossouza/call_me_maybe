from llm_sdk import Small_LLM_Model
from src.helpers import llm_vocab_load, vocab_filter_funcsname_prefix, get_instruction_funcname
from src.structures import Token
from src.callme_files_loader import CallMeFunction


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    def decode_func_name(self, prompt: str, func_names: set[str],
                         func_defs: dict[str, CallMeFunction]) -> str | None:
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
                return None
            best_token: Token = max(
                valid_tokens, key=lambda token: logits[token.id]
            )
            decoded_func += best_token.str
            if decoded_func in func_names:
                return decoded_func
