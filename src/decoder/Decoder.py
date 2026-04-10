from llm_sdk import Small_LLM_Model
from src.helpers import llm_vocab_load, vocab_filter_funcsname_prefix
from src.structures import Token


class Decoder:
    """
    Implements decoding to generate the most probable outputs using a LLM
    """

    def __init__(self) -> None:
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[str, int] = llm_vocab_load(self.llm)

    def decode_func_name(self, prompt: str,
                         func_options: set[str]) -> str | None:
        llm = self.llm
        input_ids: list[int] = llm.encode(prompt).tolist()[0]
        func_name: str = "fn_"

        print(f"\n\n{prompt}")
        while True:
            logits: list[float] = llm.get_logits_from_input_ids(input_ids)
            valid_tokens: list[Token] = vocab_filter_funcsname_prefix(
                self.vocab, func_options, func_name
            )
            if not valid_tokens:
                return None
            best_token: Token = max(
                valid_tokens, key=lambda token: logits[token.id]
            )
            func_name += best_token.str
            print(f"{func_name}: {valid_tokens}")
            if func_name in func_options:
                return func_name
            input_ids.append(best_token.id)
