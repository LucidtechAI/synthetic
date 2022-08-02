import abc

from ..core.synthesizer import Synthesizer


class ImageSynthesizer(Synthesizer):
    def __init__(self, ground_truth: list[dict]):
        super().__init__(ground_truth)

    @abc.abstractmethod
    def modify_text(self, text: str, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class BasicSynthesizer(Synthesizer):
    def __init__(self, ground_truth: list[dict]):
        super().__init__(ground_truth)

    @abc.abstractmethod
    def modify_text(self, text: str, **kwargs):
        return 'FOOBAR'

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

