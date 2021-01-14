
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionRecurrentDataset(SimpleFunctionDataset):
    def __init__(self, operation, input_size=10, **kwargs):
        super().__init__(operation, input_size=input_size, **kwargs)

    def fork(self, seq_length=10, input_range=[1, 2], *args, **kwargs):
        return super().fork((seq_length, self._input_size), input_range, *args, **kwargs)
