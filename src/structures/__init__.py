from pydantic import BaseModel


class Token(BaseModel):
    """Token structure with str and id"""
    str: str
    id: int
