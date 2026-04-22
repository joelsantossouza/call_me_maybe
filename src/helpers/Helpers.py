import json
import re
from llm_sdk import Small_LLM_Model
from src.structures import Token
from src.callme_files_loader import CallMeFunction


def llm_vocab_load(llm: Small_LLM_Model) -> dict[str, int]:
    """Get the LLM vocab json path and convert into a dictionary"""
    vocab_path: str = llm.get_path_to_vocab_file()

    with open(vocab_path, "r") as vocab:
        return json.load(vocab)


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


def extract_numbers(text: str) -> list[str]:
    """
    Return a list of string containing all numbers present
    on text
    """
    return re.findall(r"-?\d+\.?\d*", text)

def extract_names(text: str) -> list[str]:
    """
    Return all words closest to names in text
    """
    words: list[str] = re.findall(r"\b[a-zA-Z]+\b", text)
    stopwords: list[str] = [
        "the", "and", "what", "is", "to", "of",
        "greet", "times", "never", "forget", "too",
        "a"
    ]
    return [word for word in words if word.lower() not in stopwords]


def extract_strings(text: str) -> list[str]:
    """
    Return all substrings inside single or double quotes
    """
    return re.findall(r"['\"](.*?)['\"]", text)


def extract_nouns(text: str) -> list[str]:
    """
    Return all words closest to nouns in text
    """
    strings: list[str] = extract_strings(text)
    text_without_quotes: str = re.sub(r"['\"].*?['\"]", "", text)
    return extract_names(text_without_quotes) + strings


def get_instruction_funcname(
        prompt: str, func_defs: dict[str, CallMeFunction]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function name
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


def get_instruction_funcparam_number(prompt: str,
                                     func_def: CallMeFunction,
                                     param: str,
                                     nbr_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type number
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You are a strict parameter selector.\n"
        "Your job is to select the MOST appropriate numeric value for a "
        "function parameter.\n\n"
        "RULES:\n"
        "1. You MUST select EXACTLY ONE value from the provided options.\n"
        "2. You MUST NOT generate new numbers.\n"
        "3. You MUST NOT modify any number.\n"
        "4. You MUST choose the number that best matches the user intent.\n"
        "5. If multiple numbers exist, prefer those directly involved in "
        "the operation.\n"
        "6. Ignore unrelated numbers (e.g., counts, indices, or "
        "descriptions).\n"
        "7. Your output MUST be only the selected number.\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Description: {func_def.description}\n"
        f"All Parameters: {all_params}\n"
        f"Parameter to fill: {param}\n"
        f"Available options: {nbr_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def get_instruction_funcparam_string(prompt: str,
                                     func_def: CallMeFunction,
                                     param: str,
                                     string_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type string
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You are a strict parameter selector for STRING values.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. You MUST select EXACTLY ONE string from the provided options.\n"
        "2. You MUST NOT create or modify strings.\n"
        "\n"
        "PRIORITY ORDER:\n"
        "1. ALWAYS prefer strings that appear inside quotes ('...' or \"...\").\n"
        "2. If multiple quoted strings exist, choose the one that best matches the parameter role.\n"
        "3. If no quoted strings exist, choose the most relevant word or phrase from the prompt.\n"
        "\n"
        "CONTEXT RULES:\n"
        "4. If the function is about string manipulation (reverse, replace, substitute),\n"
        "   the target string is usually inside quotes.\n"
        "5. Ignore unrelated words outside quotes unless no quoted strings exist.\n"
        "\n"
        "OUTPUT RULE:\n"
        "6. Output ONLY the selected string.\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Description: {func_def.description}\n"
        f"Parameters: {all_params}\n"
        f"Parameter to fill: {param}\n"
        f"Available options: {string_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def get_instruction_funcparam_name(prompt: str,
                                     func_def: CallMeFunction,
                                     param: str,
                                     string_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type string
    and is more probable to be a name
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You are a strict parameter selector for PERSON NAMES.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. You MUST select EXACTLY ONE value from the available options.\n"
        "2. You MUST NOT create or modify names.\n"
        "\n"
        "NAME SELECTION RULES:\n"
        "3. A valid name is a word that refers to a person (e.g., 'John', 'Alice', 'Shrek').\n"
        "4. Prefer words that are targets of actions (e.g., 'greet John' → 'John').\n"
        "5. Prefer words that appear after verbs like 'greet', 'call', 'message', 'send'.\n"
        "6. Ignore generic words like 'string', 'word', 'number', 'times', etc.\n"
        "7. If multiple names exist, follow the order they appear in the prompt.\n"
        "\n"
        "CONTEXT RULES:\n"
        "8. If the prompt says 'greet X and Y', then:\n"
        "   - first parameter → X\n"
        "   - second parameter → Y\n"
        "\n"
        "9. Names may be lowercase or uppercase.\n"
        "\n"
        "OUTPUT RULE:\n"
        "10. Output ONLY the selected name.\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Description: {func_def.description}\n"
        f"Parameters: {all_params}\n"
        f"Parameter to fill: {param}\n"
        f"Available options: {string_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def get_instruction_funcparam_nouns(prompt: str,
                                     func_def: CallMeFunction,
                                     param: str,
                                     string_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type string
    and is more probable to be a noun
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You are a strict parameter selector for NOUN values.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. You MUST select EXACTLY ONE value from the available options.\n"
        "2. You MUST NOT create or modify words.\n"
        "\n"
        "NOUN SELECTION RULES:\n"
        "3. A noun is an object, entity, or thing referenced in the prompt.\n"
        "4. Prefer nouns that are DIRECTLY involved in the operation.\n"
        "5. If the prompt is about string manipulation (replace, reverse, substitute),\n"
        "   the noun is usually the TARGET word or phrase being manipulated.\n"
        "6. Ignore verbs (e.g., 'replace', 'reverse', 'calculate').\n"
        "7. Ignore generic words like 'string', 'word', 'numbers', 'thing'.\n"
        "\n"
        "CONTEXT RULES:\n"
        "10. If multiple nouns exist, select the one most relevant to the parameter role.\n"
        "11. If the parameter represents a target or input, choose the noun being acted upon.\n"
        "\n"
        "OUTPUT RULE:\n"
        "12. Output ONLY the selected value.\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Description: {func_def.description}\n"
        f"Parameters: {all_params}\n"
        f"Parameter to fill: {param}\n"
        f"Available options: {string_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
