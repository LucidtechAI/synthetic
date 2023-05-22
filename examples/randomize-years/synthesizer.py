import random
import copy
from typing import Dict, List

from synthetic.core.ground_truth import parse_labels_in_ground_truth
from synthetic.pdf.synthesizer import PdfSynthesizer
from synthetic.pdf.utils import Font


class ExcludeWordsSynthesizer(PdfSynthesizer):
    """
    This synthesizer will only randomize the years in date fields
    """
    YEARS = [str(n) for n in range(2023, 2030)]
    DATE_FIELDS = ['due_date', 'invoice_date']
    def __init__(self, ground_truth: List[dict], font_map: Dict[str, Font]):
        super().__init__(ground_truth, font_map)
        self.year_map = {}
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            if label in self.DATE_FIELDS:
                if value not in self.year_map:
                    self.year_map[value] = random.choice(self.YEARS)

    def reset(self):
        # No need to create a new substitution map, all we need is a new year for the date fields
        self.year_map = {year: random.choice(self.YEARS) for year in self.year_map}

    def modify_text(self, text: str, **kwargs):
        for original_year, new_year in self.year_map.items():
            text = text.replace(original_year, new_year)
        return text

    def create_new_ground_truth(self):
        ground_truth = copy.deepcopy(self.ground_truth)
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            year = value[:4]
            match.context.value.update({'value': value.replace(year, self.year_map[year])})
        return ground_truth