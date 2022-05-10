import abc


class Synthesizer:
    def __init__(self, ground_truth: list[dict]):
        self.ground_truth = ground_truth

    @abc.abstractmethod
    def modify_text(self, text: str, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def create_new_ground_truth(self):
        raise NotImplementedError
