from pydantic import BaseModel
from typing import List

class similar_app(BaseModel):
    AppName: str
    download_URL: str
    
class SimilarAppList(BaseModel):
    apps: List[similar_app]