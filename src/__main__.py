import json
from argparse import ArgumentParser, Namespace
from .callme_files_loader import CallMeFilesLoader
from .decoder import Decoder

if __name__ == "__main__":
    # Process CLI arguments
    parser: ArgumentParser = ArgumentParser(
        description="Call Me Maybe arguments loader",
        usage="uv run python -m src "
        "[-h] "
        "[--functions_definition <function_definition_file>] "
        "[--input <input_file>]"
        "[--output <output_file>]",
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Custom function definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Custom function prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Custom output results JSON file"
    )
    args: Namespace = parser.parse_args()

    # Load JSON files data into objects memory
    loader: CallMeFilesLoader = CallMeFilesLoader()
    try:
        with open(args.functions_definition, "r") as functions_definition_file:
            data: any = json.load(functions_definition_file)
            loader.load_functions(data)
        with open(args.input, "r") as input_file:
            data: any = json.load(input_file)
            loader.load_prompts(data)
    except Exception as error_msg:
        print(f"Error: {error_msg}")

    # Choose best function name
    decoder: Decoder = Decoder()

    for prompt in loader.prompts:
        func_name: str = decoder.decode_func_name(
            prompt.prompt, loader.func_names, loader.func_definitions
        )

        # Fill the function parameters value
        decoder.decode_func_params(
            prompt.prompt, loader.func_definitions[func_name]
        )
