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
    Return a list of strings representing numbers in float format
    """
    matches = re.findall(r"-?\d+\.?\d*", text)
    return [str(float(num)) for num in matches]


def extract_names(text: str) -> list[str]:
    """
    Return all words closest to names in text
    """
    words: list[str] = re.findall(r"[a-zA-Z]\S*", text)
    stopwords: list[str] = [
        "the", "and", "what", "is", "to", "of",
        "in", "on", "times", "never", "forget", "too",
        "a", "with"
    ]
    return [word for word in words if word.lower() not in stopwords]


def extract_strings(text: str) -> list[str]:
    """
    Return all words closest to substrings in text
    """

    # 1. quoted strings
    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', text)
    quoted = [m[0] or m[1] for m in quoted]

    additional = (
        re.findall(r"/[^\s\"']+", text) +
        re.findall(r"[A-Za-z]:\\[^\s\"']+", text)
    )

    return list(dict.fromkeys(quoted + additional))


def extract_nouns(text: str) -> list[str]:
    """
    Return all words closest to nouns in text,
    excluding already extracted quoted strings.
    """

    strings: list[str] = extract_strings(text)
    text_without_strings = text
    for s in strings:
        if s:
            text_without_strings = text_without_strings.replace(f'"{s}"', "")
            text_without_strings = text_without_strings.replace(f"'{s}'", "")
            text_without_strings = text_without_strings.replace(s, "")

    return extract_names(text_without_strings) + strings


def extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords/nouns from text for function matching.
    Focuses on action words and key concepts.
    """
    # Convert to lowercase for matching
    text_lower = text.lower()

    # Remove punctuation, treat underscores as separators, and split into words
    words = re.findall(r'[a-z]+', text_lower)

    # Common stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall', 'what', 'how',
        'when', 'where', 'why', 'which', 'who', 'that', 'this', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    }

    # Keep only meaningful keywords
    keywords = [
        word for word in words if word not in stopwords and len(word) > 2]

    # Also include short quoted strings (single words or short phrases)
    # but exclude long strings that are entire sentences
    quoted_strings = extract_strings(text)
    short_quoted = [s.lower() for s in quoted_strings if len(
        s.split()) <= 3]  # Max 3 words
    keywords.extend(short_quoted)

    return list(set(keywords))  # Remove duplicates


def get_instruction_funcname(
        prompt: str, func_defs: dict[str, CallMeFunction]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function name
    """

    # Build dynamic function list with descriptions
    avail_funcs: list[str] = [
        f"{func_name}: {func_def.description}"
        for func_name, func_def in func_defs.items()
    ]
    funcs_list: str = "\n".join(avail_funcs)

    return (
        "<|im_start|>system\n"
        "You are a strict function selector. Your job is to choose the "
        "BEST function that matches the user's intent.\n\n"
        "CORE RULES:\n"
        "1. Analyze the prompt to understand what the user wants to do.\n"
        "2. Match the user intent to the function that best fulfills it.\n"
        "3. Consider function descriptions carefully.\n"
        "4. If the prompt intent doesn't match ANY function, return "
        "'fn_none'.\n"
        "5. Prefer exact keyword matches in function names or descriptions.\n"
        "6. Do NOT guess or assume - match only what is explicitly stated "
        "in the prompt.\n"
        "7. Output ONLY the function name (e.g., 'fn_add_numbers'), "
        "nothing else.\n\n"
        "MATCHING STRATEGY:\n"
        "- Keywords in prompt → Look for matching function names or "
        "description keywords\n"
        "- Mathematical operations (multiply, divide, add, subtract, "
        "power, compound, interest, etc.) → Match with math functions\n"
        "- String operations (reverse, format, read, execute, query, etc.) "
        "→ Match with string/data functions\n"
        "- Type checks (even, odd, prime, etc.) → Match with boolean "
        "functions\n\n"
        "Available Functions:\n"
        f"{funcs_list}\n\n"
        "Remember: Choose the function that BEST matches the prompt intent. "
        "Output ONLY the function name.\n"
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
        "5. If multiple quoted strings exist and the parameter is 'source_string',\n"
        "   choose the quoted text that appears after the word 'in'.\n"
        "6. Ignore unrelated words outside quotes unless no quoted strings exist.\n"
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


def get_instruction_funcparam_regex(prompt: str,
                                    func_def: CallMeFunction,
                                    param: str,
                                    string_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type string
    and is more probable to be a regex
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You select EXACTLY ONE option representing a REGEX/PATTERN.\n"
        "\n"
        "RULES:\n"
        "- Select the value that comes BEFORE 'with' in the prompt.\n"
        "- In patterns like 'Substitute the word [VALUE] with ...', select [VALUE].\n"
        "- In patterns like 'Replace all [VALUE] with ...', select [VALUE], NOT 'all'.\n"
        "- Never select: 'word', 'all', 'every', 'each', 'any', 'Substitute', 'Replace'.\n"
        "- The regex is always the quoted text or noun that appears after 'the word' or after quantifiers.\n"
        "\n"
        "CONCRETE EXAMPLES:\n"
        "- 'Replace all numbers with NUMBERS'\n"
        "  The value BEFORE 'with' is: numbers\n"
        "  Select: 'numbers'\n"
        "\n"
        "- 'Replace all vowels with asterisks'\n"
        "  The value BEFORE 'with' is: vowels\n"
        "  Select: 'vowels'\n"
        "\n"
        "- 'Substitute the word 'cat' with 'dog' in text'\n"
        "  The value BEFORE 'with' is: 'cat' (quoted after 'the word')\n"
        "  Select: 'cat'\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Target parameter: {param}\n"
        f"Available options: {string_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_instruction_funcparam_replacement(prompt: str,
                                          func_def: CallMeFunction,
                                          param: str,
                                          string_options: list[str]) -> str:
    """
    Returns the Builded prompt to guide the LLM to
    choose the best tokens of a function parameter of type string
    and is more probable to be a replacement
    """
    all_params: dict[str, str] = {}
    for param_name, param_value in func_def.parameters.items():
        all_params[param_name] = param_value.type

    return (
        "<|im_start|>system\n"
        "You select EXACTLY ONE option representing a REPLACEMENT value.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. You MUST choose ONLY from the available options.\n"
        "2. You MUST NOT create or modify values.\n"
        "\n"
        "STRICT SELECTION LOGIC:\n"
        "3. The replacement is ALWAYS the value that comes AFTER the keyword 'with'.\n"
        "4. If a quoted string appears after 'with', it is ALWAYS the correct answer.\n"
        "5. If no quoted string exists, select the FIRST meaningful word after 'with'.\n"
        "\n"
        "HARD CONSTRAINTS:\n"
        "6. NEVER select words that appear BEFORE 'with'.\n"
        "7. NEVER select generic words such as: 'word', 'string', 'text'.\n"
        "8. NEVER select instruction words such as: 'Substitute', 'Replace'.\n"
        "9. NEVER select connector words such as: 'the', 'a', 'in', 'with'.\n"
        "\n"
        "POSITION PRIORITY:\n"
        "10. Words immediately after 'with' have MAXIMUM priority.\n"
        "11. Words far from 'with' are VERY UNLIKELY to be correct.\n"
        "\n"
        "EXAMPLES:\n"
        "- 'Replace all numbers with NUMBERS'\n"
        "  → select: NUMBERS\n"
        "\n"
        "- 'Replace all vowels with asterisks'\n"
        "  → select: asterisks\n"
        "\n"
        "- 'Substitute the word 'cat' with 'dog' in text'\n"
        "  → select: dog\n"
        "\n"
        "FINAL RULE:\n"
        "12. If a quoted value exists after 'with', ALWAYS choose it.\n"
        "\n"
        f"Function: {func_def.name}\n"
        f"Target parameter: {param}\n"
        f"Available options: {string_options}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
