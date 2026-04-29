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
            if "replace" in param:
                opts: list[str] = extract_nouns(prompt)
                return (
                    get_instruction_funcparam_replacement(
                        prompt, func_def, param, opts
                    ), opts
                )
            opts: list[str] = extract_strings(prompt)
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

    def decode_func_name(self, prompt: str, func_names: set[str],
                         func_defs: dict[str, CallMeFunction]) -> str:
        """
        Score each candidate function using keyword matching between prompt
        and function descriptions. This is more reliable than pure LLM scoring.
        """
        # Extract keywords from the prompt
        prompt_keywords: set[str] = set(extract_keywords(prompt))

        # Add fn_none as fallback
        candidates: set[str] = func_names.copy()
        candidates.add("fn_none")

        def score_function(func_name: str) -> tuple[float, float]:
            """
            Score a function using keyword overlap and LLM confidence.
            Returns (keyword_score, llm_score) for tie-breaking.
            """
            if func_name == "fn_none":
                # fn_none gets zero score for keywords, low LLM score
                return 0.0, -float('inf')

            func_def: CallMeFunction = func_defs.get(func_name)
            if not func_def:
                return 0.0, -float('inf')

            # Extract keywords from function name and description
            func_name_keywords: list[str] = extract_keywords(func_name.replace('fn_', ' '))
            desc_keywords: set[str] = set(
                extract_keywords(func_def.description) + func_name_keywords
            )

            # Calculate keyword overlap score (Jaccard similarity)
            intersection = len(prompt_keywords & desc_keywords)
            union = len(prompt_keywords | desc_keywords)
            keyword_score = intersection / union if union > 0 else 0.0

            # Also get LLM score as tiebreaker
            try:
                instruction: str = get_instruction_funcname(prompt, func_defs)
                func_with_desc: str = f"{func_name}: {func_def.description}"
                full_text: str = instruction + func_with_desc
                ids: list[int] = self.llm.encode(full_text).tolist()[0]
                logits: list[float] = self.llm.get_logits_from_input_ids(ids)
                llm_score = sum(logits) / len(logits) if logits else -float('inf')
            except Exception:
                llm_score = -float('inf')

            return keyword_score, llm_score

        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            keyword_score, llm_score = score_function(candidate)
            scored_candidates.append((candidate, keyword_score, llm_score))

        # Sort by keyword score first, then LLM score as tiebreaker
        scored_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # If the top match has a reasonable keyword score (>0.05), use it directly
        # This avoids expensive LLM calls for clear matches
        top_match = scored_candidates[0]
        if top_match[1] > 0.05:  # Jaccard score > 0.05 (5% overlap)
            return top_match[0]

        # For low-confidence matches, fall back to LLM scoring but make it faster
        # by only scoring the top 2-3 candidates by keyword score
        top_candidates = [c[0] for c in scored_candidates[:3]]  # Top 3 by keywords
        llm_scores = []
        for candidate in top_candidates:
            try:
                if candidate == "fn_none":
                    llm_scores.append((candidate, -float('inf')))
                    continue

                func_def = func_defs.get(candidate)
                if func_def:
                    instruction = get_instruction_funcname(prompt, func_defs)
                    func_with_desc = f"{candidate}: {func_def.description}"
                    full_text = instruction + func_with_desc
                    ids = self.llm.encode(full_text).tolist()[0]
                    logits = self.llm.get_logits_from_input_ids(ids)
                    score = sum(logits) / len(logits) if logits else -float('inf')
                    llm_scores.append((candidate, score))
                else:
                    llm_scores.append((candidate, -float('inf')))
            except Exception:
                llm_scores.append((candidate, -float('inf')))

        llm_scores.sort(key=lambda x: x[1], reverse=True)
        return llm_scores[0][0]
