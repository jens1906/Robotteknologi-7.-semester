"""KPR person re-identification, inference only (see NOTICE.md)."""
from kpr.extractor import Kpr, pick_device
from kpr.match import part_distance
from kpr.prompt import PART_NAMES

__all__ = ['Kpr', 'part_distance', 'pick_device', 'PART_NAMES']
