import random
import re
import copy
from datetime import datetime
from typing import Dict, List

from synthetic.core.ground_truth import parse_labels_in_ground_truth
from synthetic.pdf.synthesizer import PdfSynthesizer
from synthetic.pdf.utils import Font


# Courtesy of https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729
def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.

    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str

    """
    if not replacements:
        return string
    # Place longer ones first to keep shorter substrings from matching
    # where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against
    # the string 'hey abc', it should produce 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


class RandomizeYearsSynthesizer(PdfSynthesizer):
    """
    This synthesizer will only randomize the years in date fields
    """
    YEARS = [str(n) for n in range(2016, 2029) if n != 2020]
    DATE_FIELDS = ['due_date', 'invoice_date']
    def __init__(self, ground_truth: List[dict], font_map: Dict[str, Font]):
        super().__init__(ground_truth, font_map)
        self.year_map = {}
        self.post_year_map = {}
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            if label in self.DATE_FIELDS:
                year = str(datetime.strptime(value.strip(), '%Y-%m-%d').year)
                if year not in self.year_map:
                    dst_year = random.choice(self.YEARS)
                    self.year_map[year[2:]] = dst_year[2:]
                    if year == '2020':
                        self.post_year_map[dst_year[2:]*2] = dst_year

    def reset(self):
        # No need to create a new substitution map, all we need is a new year for the date fields
        self.year_map = {year: random.choice(self.YEARS)[2:] for year in self.year_map}

    def modify_text(self, text: str, **kwargs):
        return multireplace(multireplace(text, self.year_map), self.post_year_map)

    def create_new_ground_truth(self):
        ground_truth = copy.deepcopy(self.ground_truth)
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            match.context.value.update({'value': multireplace(multireplace(value, self.year_map), self.post_year_map)})
        return ground_truth
