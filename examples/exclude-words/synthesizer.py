import copy
import json
import pathlib

from synthetic.core.ground_truth import parse_labels_in_ground_truth
from synthetic.pdf.synthesizer import BasicSynthesizer


class ExcludeWordsSynthesizer(BasicSynthesizer):
    """
    This synthesizer ignores synthesizing words that appear in EXCLUDED_WORDS (./words.json file)
    """
    WORDS_DELIM = ' '
    WORDS_FILE = pathlib.Path(__file__).parent / 'words.json'
    CAPITALIZE_WORDS = True
    EXCLUDED_WORDS = set(json.loads(WORDS_FILE.read_text()))

    def modify_text(self, text: str, **kwargs):
        return self._split_and_modify(text)

    def create_new_ground_truth(self):
        ground_truth = copy.deepcopy(self.ground_truth)
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            if isinstance(value, str):
                match.context.value.update({'value': self._split_and_modify(value)})
            elif isinstance(value, float) or isinstance(value, int):
                match.context.value.update({'value': self._split_and_modify(str(value))})
        return ground_truth

    def _split_and_modify(self, text):
        words = []
        for word in text.split(self.WORDS_DELIM):
            if (word.upper() if self.CAPITALIZE_WORDS else word) in self.EXCLUDED_WORDS:
                words.append(word)
            else:
                words.append(self.substitute(word))
        return self.WORDS_DELIM.join(words)
