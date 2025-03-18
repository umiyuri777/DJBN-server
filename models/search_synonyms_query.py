from pydantic import BaseModel

class search_synonyms_query(BaseModel):
    query: str