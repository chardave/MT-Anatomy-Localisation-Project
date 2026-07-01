from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ScanRecord:
    case_id: str
    dataset: str
    image_path: Path
    label_path: Optional[Path]
    anatomical_region: str
    labels_present: List[str]
    annotation_type: str  # "manual", "automatic", "mixed", "unknown"