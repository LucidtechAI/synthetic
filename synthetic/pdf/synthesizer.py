import abc
import copy
import itertools
import random
import string
from functools import reduce

from ..core.ground_truth import parse_labels_in_ground_truth
from ..core.synthesizer import Synthesizer
from .utils import Font


class PdfSynthesizer(Synthesizer):
    def __init__(self, ground_truth: list[dict], font_map: dict[str, Font]):
        super().__init__(ground_truth)
        self.font_map = font_map

    @abc.abstractmethod
    def modify_text(self, text: str, **kwargs):
        raise NotImplementedError


class BasicSynthesizer(PdfSynthesizer):
    def __init__(
        self,
        ground_truth: list[dict],
        font_map: dict[str, Font],
    ):
        super().__init__(ground_truth, font_map)
        characters_groups = [font.available_characters() for font in self.font_map.values()]
        self.substitution_map = self._create_substitution_map(characters_groups, string.ascii_letters)

    def modify_text(self, text: str, **kwargs):
        return self.substitute(text)

    def create_new_ground_truth(self):
        for label, value, match in parse_labels_in_ground_truth(copy.deepcopy(self.ground_truth)):
            if isinstance(value, str):
                match.context.value.update({'value': self.substitute(value)})
        return self.ground_truth

    def substitute(self, text):
        return ''.join(self.substitution_map.get(c, c) for c in text)

    def _create_substitution_map(self, character_groups, characters_to_substitute):
        def remap(character_group):
            if characters_to_substitute:
                return set(characters_to_substitute) & set(character_group)
            else:
                return set(character_group)

        def not_remap(character_group):
            return set(character_group) - remap(character_group)

        characters_to_remap = [remap(character_group) for character_group in character_groups]
        characters_to_not_remap = reduce(lambda a, b: a | b, map(not_remap, character_groups))
        number_of_fonts = len(self.font_map)
        substitution_map = {c: c for c in characters_to_not_remap}

        for i in range(number_of_fonts):
            partial_mapping = {}

            for available_characters in itertools.combinations(characters_to_remap, number_of_fonts - i):
                intersection = list(reduce(lambda a, b: a & b, available_characters))
                partial_mapping.update(create_substitutions(intersection, intersection))

            for available_characters in characters_to_remap:
                for c in partial_mapping:
                    available_characters.discard(c)

            substitution_map.update(partial_mapping)

        return substitution_map


def create_substitutions(all_characters, characters_to_substitute):
    from_characters = all_characters[:]
    indices_to_shuffle = [i for i, c in enumerate(all_characters) if c in characters_to_substitute]
    to_characters = shuffle_indices(from_characters, indices_to_shuffle)
    return {k: v for k, v in zip(from_characters, to_characters)}


def shuffle_indices(sequence, indices_to_shuffle):
    shuffled_indices = indices_to_shuffle[:]
    random.shuffle(shuffled_indices)

    index_map = {}
    for from_index, to_index in zip(indices_to_shuffle, shuffled_indices):
        index_map[from_index] = to_index

    new_sequence = []
    for i, x in enumerate(sequence):
        if i in index_map:
            new_sequence.append(sequence[index_map[i]])
        else:
            new_sequence.append(x)
    return new_sequence
