from typing import List
from pydantic import BaseModel

class Level(BaseModel):
        level: int
        title: str

class ToC(BaseModel):
    structure: List[Level]