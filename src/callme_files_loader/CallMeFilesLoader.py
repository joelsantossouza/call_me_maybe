from . import CallMeFunction, CallMePrompt


class CallMeFilesLoader:
    """
    Class to validate and load the Call Me Maybe files
    into usable structures
    """

    def __init__(self) -> None:
        self.functions: dict[str, CallMeFunction] = {}
        self.prompts: list[str] = []

    def load_functions(self, file: list[dict]) -> None:
        for function in file:
            validated_function: CallMeFunction = CallMeFunction(**function)
            self.functions[validated_function.name] = validated_function

    def load_prompts(self, file: list[dict]) -> None:
        for prompt in file:
            validated_prompt: CallMePrompt = CallMePrompt(**prompt)
            self.prompts.append(validated_prompt)
