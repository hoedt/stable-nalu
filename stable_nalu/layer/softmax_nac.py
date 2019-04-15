
import math
import torch

from ..abstract import ExtendedTorchModule
from ._abstract_recurrent_cell import AbstractRecurrentCell

class SoftmaxNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        # Define the target weights. Also, put 0 last such that p1 = p2 = 0
        # corresponds to p3 = 1 => w = 0.
        self.register_buffer('target_weights', torch.tensor([1, -1, 0], dtype=torch.float32))

        # We will only two parameters per weight, this is to prevent the redundancy
        # there would otherwise exist. This also makes it much more comparable with
        # NAC.
        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features, 2))
        self.register_buffer('W_hat_k', torch.Tensor(out_features, in_features, 1))

        self.register_parameter('bias', None)

    def reset_parameters(self):
        # Use a gain of sqrt(0.5). Lets assume that softmax'(0) ~ 1, because this
        # holds for sigmoid. Then:
        #   Var[W] = 1 * Var[S_1] - 1 * Var[S_2] + 0 * Var[S_3] = 2 / (fan[in] + fan[out])
        #   Var[W] = 2 * Var[S_i] = 2 / (fan[in] + fan[out])
        #   Var[S_i] = 1/2 * 2 / (fan[in] + fan[out])
        #   sqrt(Var[S_i]) = sqrt(1/2) * sqrt(2 / (fan[in] + fan[out]))
        # This is not exactly true, because S_1, S_2, and S_3 are not enterily uncorrelated.
        torch.nn.init.xavier_uniform_(self.W_hat, gain=math.sqrt(0.5))
        torch.nn.init.constant_(self.W_hat_k, 0)

    def forward(self, input, reuse=False):
        # Concat trainable and non-trainable weights
        W_hat_full = torch.cat((self.W_hat, self.W_hat_k), dim=-1)  # size = [out, in, 3]

        # Compute W
        pi = torch.nn.functional.softmax(W_hat_full, dim=-1)
        W = pi @ self.target_weights

        # Compute the linear multiplication as usual
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class SoftmaxNACCell(AbstractRecurrentCell):
    """Implements the Gumbel NAC (Gumbel Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(SoftmaxNACLayer, input_size, hidden_size, **kwargs)
