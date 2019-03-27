
import numpy as np
import torch
import torch.utils.data

class ARITHMETIC_FUNCTIONS:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    def div(a, b):
        return a / b

    def squared(a, b):
        return a * a

    def root(a, b):
        return np.sqrt(a)

class SimpleFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, operation, shape,
                 input_range=1,
                 seed=None,
                 num_workers=1,
                 max_size=2**32-1):
        super().__init__()

        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._input_range = input_range
        self._shape = shape
        self._input_size = shape[-1]
        self._rngs = [
            np.random.RandomState(None if seed is None else seed + i)
            for i in range(max(1, num_workers))
        ]
        self._worker_id = 0

        self.a_start = self._rngs[0].randint(0, self._input_size)
        a_size = self._rngs[0].randint(1, self._input_size - self.a_start + 1)
        self.a_end = self.a_start + a_size

        self.b_start = self._rngs[0].randint(0, self._input_size)
        b_size = self._rngs[0].randint(1, self._input_size - self.b_start + 1)
        self.b_end = self.b_start + b_size

    def _worker_init_fn(self, worker_id):
        self._worker_id = worker_id

    def __getitem__(self, index):
        input_vector = self._rngs[self._worker_id].uniform(
            low=0,
            high=self._input_range,
            size=self._shape)

        # Compute a and b values
        a = np.sum(input_vector[..., self.a_start:self.a_end])
        b = np.sum(input_vector[..., self.b_start:self.b_end])
        # Compute result of arithmetic operation
        output_scalar = self._operation(a, b)

        return (
            torch.tensor(input_vector, dtype=torch.float32),
            torch.tensor([output_scalar], dtype=torch.float32)
        )

    def __len__(self):
        return self._max_size

    @classmethod
    def dataloader(cls, operation=None, batch_size=128, num_workers=0, use_cuda=False, **kwargs):
        print(cls)
        if operation is None:
            raise ValueError('operation must be specified')

        dataset = cls(operation, num_workers=num_workers, **kwargs)
        batcher = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(dataset),
            num_workers=num_workers,
            worker_init_fn=dataset._worker_init_fn)

        if use_cuda:
            return map(lambda X, t: (X.cuda(), t.cuda()), batcher)
        else:
            return batcher
