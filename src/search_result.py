from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SearchResult:
    id: int
    distance: float

    kwarg_dict: Optional[Dict[str, Any]]