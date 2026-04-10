import json
from llm_sdk import Small_LLM_Model
from src.structures import Token


def llm_vocab_load(llm: Small_LLM_Model) -> dict[str, int]:
    """Get the LLM vocab json path and convert into a dictionary"""
    vocab_path: str = llm.get_path_to_vocab_file()

    with open(vocab_path, "r") as vocab:
        return json.load(vocab)


def token_list_insert_sorted(tokens: list[Token],
                             insert_token: Token) -> None:
    """
    Insert new token in tokens, sorting by descending order
    based on Token.str length
    """
    i: int = 0
    insert_token_len: int = len(insert_token.str)

    for token in tokens:
        if len(token.str) < insert_token_len:
            break
        i += 1
    tokens.insert(i, insert_token)


def vocab_filter_funcsname_prefix(vocab: dict[str, int], funcsname: set[str],
                                  prefix: str) -> list[Token]:
    """
    Returns vocabulary tokens that can still form a valid
    function name given the current prefix
    """
    tokens_filtered: list[Token] = []

    for token_str, token_id in vocab.items():
        candidate: str = prefix + token_str
        if any(func.startswith(candidate) for func in funcsname):
            tokens_filtered.append(
                Token(str=token_str, id=token_id)
            )
    return tokens_filtered
