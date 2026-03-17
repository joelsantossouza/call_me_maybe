from pydantic import BaseModel


class CallMeParameter(BaseModel):
    """
    Function parameter struct validation
    """
    type: str


class CallMeFunction(BaseModel):
    """
    Function definition structure validation
    """
    name: str
    description: str
    parameters: dict[str, CallMeParameter]
    returns: dict[str, str]


class CallMePrompt(BaseModel):
    """
    Prompt structure validation
    """
    data: str
