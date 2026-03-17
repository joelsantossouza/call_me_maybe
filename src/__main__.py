from argparse import ArgumentParser
from .callme_files_loader import CallMeFilesLoader

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
    args = parser.parse_args()
    print(args)

    # Load files data into objets memory
    loader: CallMeFilesLoader = CallMeFilesLoader()
