from pydantic import BaseModel
from typing import List

class Similar_app(BaseModel):
    AppName: str
    download_URL: str
    
class SimilarAppList(BaseModel):
    apps: List[Similar_app]