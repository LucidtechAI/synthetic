import abc
import copy
import random
import string
from functools import reduce
from itertools import chain, combinations
from typing import Dict, List

from ..core.ground_truth import parse_labels_in_ground_truth
from ..core.synthesizer import Synthesizer
from .utils import Font


class PdfSynthesizer(Synthesizer):
    def __init__(self, ground_truth: List[dict], font_map: Dict[str, Font]):
        super().__init__(ground_truth)
        self.font_map = font_map

    @abc.abstractmethod
    def modify_text(self, text: str, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class BasicSynthesizer(PdfSynthesizer):
    def __init__(self, ground_truth: List[dict], font_map: Dict[str, Font]):
        super().__init__(ground_truth, font_map)
        self.substitutions = self._create_substitution_map()

    def modify_text(self, text: str, **kwargs):
        return self.substitute(text)

    def reset(self):
        self.substitutions = self._create_substitution_map()

    def create_new_ground_truth(self):
        ground_truth = copy.deepcopy(self.ground_truth)
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            if isinstance(value, str):
                match.context.value.update({'value': self.substitute(value)})
            elif isinstance(value, float) or isinstance(value, int):
                match.context.value.update({'value': self.substitute(str(value))})
        return ground_truth

    def substitute(self, text):
        return ''.join(self.substitutions.get(c, c) for c in text)

    def _create_substitution_map(self):
        available_character_sets = [font.available_characters for font in self.font_map.values()]
        substitution_character_sets = [
            # Since prefixed and suffixed zeros are often stripped in amounts, we avoid synthesizing those
            set(string.digits.replace('0', '')),
            set(string.ascii_lowercase),
            set(string.ascii_uppercase),
        ]

        if len(substitution_character_sets) > 1:
            assert all([a.isdisjoint(b) for a, b in combinations(substitution_character_sets, 2)])

        list_of_character_sets = []

        for substitution_group in substitution_character_sets:
            character_sets = []
            for available_characters in available_character_sets:
                character_sets.append(available_characters & substitution_group)
            list_of_character_sets.append(character_sets)

        substitution_map = {}

        for character_sets in list_of_character_sets:
            substitution_map.update(self._remap_characters(character_sets))

        for c in chain(*available_character_sets):
            if c not in substitution_map:
                substitution_map[c] = c

        return substitution_map

    @staticmethod
    def _remap_characters(character_sets):
        substitution_map = {}
        number_of_sets = len(character_sets)

        for i in range(number_of_sets):
            partial_mapping = {}

            for available_characters in combinations(character_sets, number_of_sets - i):
                intersection = list(reduce(lambda a, b: a & b, available_characters))
                partial_mapping.update(create_substitutions(intersection, intersection))

            for available_characters in character_sets:
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
