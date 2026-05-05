*This project has been created as part of the 42 curriculum by joesanto.*

---

# call me maybe

> Introduction to function calling in LLMs — translating natural language into structured, machine-executable function calls using constrained decoding.

---

## Description

**call me maybe** is a function calling tool that bridges the gap between natural language and computer-executable code. Given a plain-English prompt such as `"What is the sum of 40 and 2?"`, the system does **not** return `42` — instead, it produces a structured JSON object:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

Large Language Models are excellent at understanding human language, but small models (≤ 1B parameters) are notoriously unreliable at producing well-formed JSON on their own — succeeding as little as 30% of the time with naive prompting. This project solves that problem through **constrained decoding**: a technique that intercepts the model's token generation at each step, masks out any token that would violate the allowed output, and forces the model to always select from a predefined set of valid strings.

The result is near-perfect reliability even with the lightweight **Qwen/Qwen3-0.6B** model (500 million parameters), demonstrating that structural guidance can outperform raw model size.

### Key features

- Constrained decoding via a **prefix trie** built from tokenised candidate strings — only tokens that advance toward a valid option can ever be selected
- **LLM-driven** function name selection and parameter value assignment (no heuristics, no hardcoding)
- Type-aware candidate extraction from prompts (`number`, `integer`, `string`)
- Deduplication and cleaning of extracted candidates to avoid trivially wrong assignments
- Fully typed Python with clean separation between loading, decoding, and helpers
- Robust error handling throughout

---

## Instructions

### Prerequisites

- Python 3.10 or later
- [`uv`](https://github.com/astral-sh/uv) package manager
- The `llm_sdk` package provided by the school (copy it into the project root)

### Installation

```bash
# Install uv and all dependencies
make install
```

### Running the program

```bash
# Run with default paths
make run

# Override any path via make variables
make run FUNCTIONS=path/to/functions_definition.json \
         INPUT=path/to/tests.json \
         OUTPUT=path/to/results.json
```

Default paths used when no variables are overridden:

| Variable | Default |
|---|---|
| `FUNCTIONS` | `./data/input/functions_definition.json` |
| `INPUT` | `./data/input/function_calling_tests.json` |
| `OUTPUT` | `./data/output/function_calls.json` |

### All Makefile targets

| Target | Description |
|---|---|
| `make install` | Install `uv` via `pip`, then sync all dependencies |
| `make run` | Run the program with the configured paths |
| `make debug` | Run under Python's `pdb` debugger |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, and `*.pyc` files |
| `make lint` | Run `flake8` + `mypy` with standard flags |
| `make lint-strict` | Run `flake8` + `mypy --strict` |

> **Note:** The virtual environment is stored at `$HOME/sgoinfre/.venv` and the uv cache at `$HOME/sgoinfre` to comply with 42 school disk quota restrictions on the home partition.

---

## Example Usage

### Input: `data/input/functions_definition.json`

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

### Input: `data/input/function_calling_tests.json`

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

### Output: `data/output/function_calls.json`

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Algorithm Explanation

The pipeline runs two sequential constrained decoding phases for each prompt: **function name selection**, then **parameter value assignment**. Both phases share the same underlying mechanism — a prefix trie — but operate over different candidate sets and use different prompts.

### Core mechanism: prefix trie constrained decoding (`constrained_decode_from_options`)

Given a list of candidate strings (e.g. function names or extracted values):

1. Each candidate is tokenised using `llm.encode`, producing a sequence of token IDs.
2. A **prefix trie** is built from all these token ID sequences. Every path through the trie from root to an `__END__` node corresponds to exactly one valid candidate.
3. The model is called with an instruction prompt and iteratively generates tokens:
   - At each step, the model produces a logit for every token in the vocabulary.
   - For every token ID, `is_valid_prefix` checks whether appending that token to the already-generated sequence still lies on a valid path in the trie. If not, its logit is set to `-inf`.
   - The token with the highest remaining logit is selected (greedy / argmax).
   - Generation stops as soon as the decoded output exactly matches one of the candidates.
4. If generation exceeds `MAX_TOKENS` (20) without a match, the first candidate is returned as a safe fallback.

This guarantees the model can only ever output one of the provided candidate strings.

### Phase 1 — Function name selection (`decode_func_name`)

- **Candidates:** all function names from `functions_definition.json`.
- **Prompt** (`build_instruction_for_func_name`): lists all function names and instructs the model to output exactly one, ending with `Function name: ` as the generation prefix.
- The LLM's own probability distribution over the constrained token set determines which function name is selected.

### Phase 2 — Parameter value assignment (`decode_func_params`)

Parameters are processed in **descending name-length order** (longest first) to minimise substring ambiguity. A `used` set tracks already-assigned values so no value is assigned to two parameters.

For each parameter:

**a) Candidate extraction** (`extract_params_options`):

The parameter's declared type routes extraction to one of three regex-based functions:

| Type | Function | What it extracts |
|---|---|---|
| `"number"` | `extract_numbers` | Integers and floats, returned as float strings (e.g. `"2.0"`) |
| `"integer"` | `extract_ints` | Integers only, excluding float parts |
| `"string"` | `extract_strings` | Quoted substrings → colon-introduced values → unquoted tokens |

`extract_strings` works in three passes: it first captures colon-introduced values (e.g. `input: hello`), then quoted strings (`'hello'`, `"hello"`), then remaining unquoted tokens that start with a non-digit character. Common stop words are filtered out.

After extraction, any candidate whose lowercased form matches the function name, its underscore-split parts, or any parameter name is removed. This prevents the model from confusing structural labels with actual values.

**b) Constrained decoding** (`constrained_decode_from_options`): same trie mechanism as Phase 1, with a richer prompt (`build_instruction_for_func_params`) that uses the Qwen3 chat template format (`<|im_start|>system ... <|im_end|>`) and includes the function description, all parameter types, the candidate list, and ends with `<param> = ` as the generation prefix.

### Prefix trie structure

```
root = {
  token_id_A: {
    token_id_B: {"__END__": True},   ← candidate "fn_greet" ends here
    token_id_C: { ... }
  },
  token_id_X: { ... }
}
```

`is_valid_prefix(candidate_so_far, trie)` traverses from root following the generated token IDs. If any token is not a child of the current node, the candidate sequence is invalid.

---

## Design Decisions

**Prefix trie on token IDs, not characters.** A trie built directly from tokenised sequences respects the tokeniser's actual segmentation. There is no need to handle the `Ġ` space-prefix convention or multi-character token fragmentation: the trie operates at the same granularity as the model's output, making `is_valid_prefix` exact and efficient.

**Two-phase decoding.** Separating function selection from parameter assignment keeps each phase simple and independently testable. The function name phase needs only function-level context; the parameter phase has access to the full function definition.

**Longest-parameter-first ordering.** When two parameters share type and the prompt contains multiple values, processing the longer-named parameter first and tracking a `used` set prevents the same value from being greedily consumed by the wrong parameter.

**Greedy (argmax) decoding.** Function calling is a deterministic extraction task. Sampling would add noise, make results non-reproducible, and complicate debugging.

**Structured Qwen3 chat-template prompts for parameter assignment.** Using `<|im_start|>` / `<|im_end|>` for parameter prompts aligns with how the model was instruction-tuned, producing sharper probability distributions over the correct candidate tokens compared to plain prompts.

**Candidate cleaning before decoding.** Removing function name fragments and parameter names from the candidate list before building the trie reduces trie size, speeds up masking, and prevents the model from selecting a label word (e.g. `"reverse"`) instead of the actual value.

---

## Performance Analysis

### Accuracy

With constrained decoding active:

- **Valid output: 100%** — the trie guarantees the decoder only emits strings from the candidate list. The output is always a known function name or an extracted parameter value.
- **Function selection: ~90–95%** — driven by the LLM's understanding of the prompt. Accuracy drops on highly ambiguous prompts where multiple functions could plausibly apply.

Without constrained decoding, JSON validity with Qwen3-0.6B drops to approximately 30%.

### Speed

- Model loading: ~10–20 seconds (first run).
- Per-prompt processing: ~3–8 seconds on CPU depending on prompt length and parameter count.
- A batch of 20 prompts: approximately 2–4 minutes on standard CPU hardware, within the 5-minute requirement.

The bottleneck is the LLM forward pass (one per generated token). The trie lookup is O(k) where k is the number of tokens generated and is negligible.

### Reliability

- If the candidate list is empty, `constrained_decode_from_options` returns `"none"` immediately.
- If a single candidate is provided, it is returned without calling the model.
- If generation exceeds `MAX_TOKENS`, the first candidate is returned as a safe fallback.
- All file I/O uses `try/except` with clear error messages; the program never crashes unexpectedly.

---

## Challenges Faced

**Token fragmentation.** A name like `fn_add_numbers` is split into multiple tokens by the tokeniser. The trie must be built from `llm.encode(name)` output — not from characters. Getting this right was the key insight that made constrained decoding work correctly.

**Extracting the right candidates.** Raw regex extraction from a prompt like `"Greet shrek"` produces `["Greet", "shrek"]`, both of which look like valid string candidates. The cleaning step (removing function-name parts and parameter names) is essential to filter out `"Greet"` and leave only `"shrek"`.

**Value contamination across parameters.** For functions with multiple parameters of the same type, without deduplication the same extracted value could be assigned to both. The `used` set and longest-first ordering together solve this.

**Prompt sensitivity.** Plain prompts for parameter assignment produced weak logit distributions. Switching to the Qwen3 chat template with an explicit system message describing the function and the candidate list noticeably improved selection accuracy on ambiguous prompts.

**`extract_strings` ordering matters.** Colon-introduced values (e.g. `input: hello`) must be captured before removing punctuation, or the colon pattern is lost. Quoted strings must be captured and removed from the text before extracting unquoted tokens, otherwise the quoted content is double-counted.

---

## Testing Strategy

### Integration tests

End-to-end runs with a small 3-function definition file and 10 prompts with known correct outputs. Post-run checks:

1. Output file is valid JSON.
2. Every entry has `prompt`, `name`, and `parameters` keys.
3. `name` values exist in the definitions file.
4. Argument types match the declared types.

### Edge cases tested

- Two parameters of the same type with distinct values in the prompt
- Integer parameter where the prompt contains a float (must not be extracted by `extract_ints`)
- Quoted string arguments, colon-introduced arguments, bare-word arguments
- Empty candidate list (should return `"none"`)
- Missing or malformed input files

---

## Resources

### Documentation and references

- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Byte-Pair Encoding tokenisation — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)
- [OpenAI function calling documentation](https://platform.openai.com/docs/guides/function-calling) — reference for the motivation and output format.
- [Python `re` module documentation](https://docs.python.org/3/library/re.html)

### AI usage

AI tools were used in the following ways during this project:

| Task | How AI was used |
|---|---|
| Debugging trie edge cases | Described token fragmentation issues to AI to understand why certain names were not decoding correctly |
| README drafting | Claude produced an initial draft which was then reviewed, corrected, and rewritten to accurately reflect our implementation |
