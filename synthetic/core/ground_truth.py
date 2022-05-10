import abc
import string
from collections import namedtuple
from functools import singledispatch

import dateparser
from babel.numbers import parse_number, parse_decimal, NumberFormatError
from jsonpath_ng import parse


class ValueType:
    def __init__(self, original_value, python_value, api_type):
        self.original_value = original_value
        self.python_value = python_value
        self.api_type = api_type

    @classmethod
    @abc.abstractmethod
    def parse(cls, original_value):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def random(cls):
        raise NotImplementedError

    def __eq__(self, other):
        return self.python_value == other.python_value

    def __repr__(self):
        return f'{self.__class__.__name__}({self.original_value})'


class StringType(ValueType):
    def __init__(self, original_value):
        super().__init__(original_value, str(original_value), 'string')

    @classmethod
    def parse(cls, original_value):
        return None

    @classmethod
    def random(cls):
        return None


class DateType(ValueType):
    SETTINGS = {
        'NORMALIZE': True,
        'STRICT_PARSING': True,
    }

    def __init__(self, original_value, python_value):
        super().__init__(original_value, python_value, 'date')

    @classmethod
    def parse(cls, original_value):
        if not 3 <= sum(c in string.digits for c in original_value) <= 8:
            return

        if python_value := dateparser.parse(original_value, settings=cls.SETTINGS):
            return cls(original_value, python_value)

    @classmethod
    def random(cls):
        return None


class AmountType(ValueType):
    def __init__(self, original_value, python_value):
        super().__init__(original_value, python_value, 'amount')

    @classmethod
    def parse(cls, original_value):
        try:
            python_value = parse_number(original_value) or parse_decimal(original_value)
            return cls(original_value, python_value)
        except NumberFormatError:
            pass

    @classmethod
    def random(cls):
        return None


class DigitsType(ValueType):
    def __init__(self, original_value, python_value):
        super().__init__(original_value, python_value, 'digits')

    @classmethod
    def parse(cls, original_value):
        if all(c in string.digits for c in original_value):
            return cls(original_value, original_value)

    @classmethod
    def random(cls):
        return None


class CurrencyType(ValueType):
    CURRENCIES = {
        'NOK',
        'EUR',
        'USD',
    }

    def __init__(self, original_value, python_value):
        super().__init__(original_value, python_value, 'string')

    @classmethod
    def parse(cls, original_value):
        if original_value in cls.CURRENCIES:
            return cls(original_value, original_value)

    @classmethod
    def random(cls):
        return None


@singledispatch
def resolve_value_types(value, api_type=None):
    raise ValueError(f'Could not resolve value types for {value} with type {type(value)}')


@resolve_value_types.register
def _(value: int, api_type=None):
    value_types = [DigitsType(value, value)]

    if api_type == 'amount':
        value_types.append(AmountType(value, value))

    return value_types


@resolve_value_types.register
def _(value: float, api_type=None):
    value_types = [AmountType(value, value)]

    if value.is_integer() or api_type == 'digits':
        value_types.append(DigitsType(value, value))

    return value_types


@resolve_value_types.register
def _(value: str, api_type=None):
    value_types = []
    if not value:
        return value_types

    if api_type == 'date':
        value_types.append(DateType.parse(value))
    elif api_type == 'digits':
        value_types.append(DigitsType.parse(value))
    elif api_type == 'amount':
        value_types.append(AmountType.parse(value))
    else:
        value_types.extend([
            AmountType.parse(value),
            DigitsType.parse(value),
            DateType.parse(value),
            CurrencyType.parse(value),
        ])

    return list(filter(bool, value_types))


def parse_labels_in_ground_truth(ground_truth):
    for match in parse('$..label').find(ground_truth):
        yield match.context.value['label'], match.context.value['value'], match


class GroundTruthMatcher:
    GroundTruthItem = namedtuple('GroundTruthItem', ['label', 'value', 'value_types', 'json_path'])

    def __init__(self, ground_truth):
        self.ground_truth_items = []
        for label, value, match in parse_labels_in_ground_truth(ground_truth):
            try:
                value_types = resolve_value_types(value, match.context.value.get('type'))
                self.ground_truth_items.append(self.GroundTruthItem(
                    label=label,
                    value=value,
                    value_types=value_types,
                    json_path=match.full_path,
                ))
            except ValueError:
                pass

    def match(self, value_type):
        def filter_fn(_value_type):
            return type(_value_type) == type(value_type)

        for ground_truth_item in self.ground_truth_items:
            for ground_truth_value_type in filter(filter_fn, ground_truth_item.value_types):
                if ground_truth_value_type == value_type:
                    yield ground_truth_item
