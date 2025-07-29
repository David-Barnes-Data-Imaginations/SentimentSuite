from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel

class SentimentSummary(BaseModel):
    emotion: str
    mean: float
    std: float
    max_val: float
    min_val: float

class AnalysisResults:
    def __init__(self):
        self.results: Dict[str, List[Union[Dict[str, Any], SentimentSummary]]] = {
            'modernbert': [],
            'bart': [],
            'nous-hermes': []
        }
        self.timestamp: Optional[datetime] = None

analysis_store = AnalysisResults()