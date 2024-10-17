from langchain_core.pydantic_v1 import BaseModel, Field

class Schema:
    class EnhancedQueries(BaseModel):
        search_query: str = Field(description="The enhanced search query")
