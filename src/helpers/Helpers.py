import json
from llm_sdk import Small_LLM_Model
from src.structures import Token
from src.callme_files_loader import CallMeFunction


def llm_vocab_load(llm: Small_LLM_Model) -> dict[str, int]:
    """Get the LLM vocab json path and convert into a dictionary"""
    vocab_path: str = llm.get_path_to_vocab_file()

    with open(vocab_path, "r") as vocab:
        return json.load(vocab)

#def llm_vocab_load(llm: Small_LLM_Model) -> dict[str, int]:
#    """Get the LLM vocab json path and convert into a dictionary"""
#    vocab_path: str = llm.get_path_to_vocab_file()
#    filtered_vocab: dict[str, int] = {}
#
#    with open(vocab_path, "r") as vocab:
#        data: any = json.load(vocab)
#        for token_str, token_id in data.items():
#            if token_str.startswith("Ġ"):
#                token_str = token_str[:1]
#            filtered_vocab[token_str] = token_id
#        return filtered_vocab
#


#def token_list_insert_sorted(tokens: list[Token],
#                             insert_token: Token) -> None:
#    """
#    Insert new token in tokens, sorting by descending order
#    based on Token.str length
#    """
#    i: int = 0
#    insert_token_len: int = len(insert_token.str)
#
#    for token in tokens:
#        if len(token.str) < insert_token_len:
#            break
#        i += 1
#    tokens.insert(i, insert_token)


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


def get_instruction_funcname(
        prompt: str, func_defs: dict[str, CallMeFunction]) -> str | None:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens
    """

    avail_funcs: list[str] = [
        f"{func_name} - {func_def.description}"
        for func_name, func_def in func_defs.items()
    ]
    return (
        "<|im_start|>system\n"
        "You are a strict function selector.\n"
        "RULES:\n"
        "1. If the prompt contains 'replace', 'with', or 'regex', "
        "you MUST use 'fn_substitute_string_with_regex'.\n"
        "2. NEVER use 'fn_execute_sql_query' unless keywords like "
        "'SELECT', 'INSERT', 'UPDATE' or 'DATABASE' are present.\n"
        "3. If no function matches the intent exactly, return 'fn_none'.\n"
        "4. NEVER use 'fn_greet' unless the prompt explicitly asks to "
        "Greet/greet or say hello to a person.\n"
        "5. NEVER use any function if the prompt asks for something "
        "none of the functions can do.\n"
        "6. NEVER use 'fn_get_square_root' unless the prompt explicitly "
        "mentions 'square root', 'sqrt', 'square root of', or "
        "'raiz quadrada'. The word 'square' alone does NOT trigger "
        "this function.\n"
        "7. NEVER use 'fn_execute_sql_query' for mathematical "
        "expressions like '^', 'power', or 'exponent'.\n"
        "8. NEVER use 'fn_calculate_compound_interest' unless the prompt "
        "explicitly mentions 'interest', 'rate', "
        "'principal', or 'compound'.\n"
        f"Available Functions:\n{avail_funcs}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
