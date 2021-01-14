import torch
from torch import nn

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.layer.mclstm import get_redistribution, Gate


class MCFullyConnected(ExtendedTorchModule):

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__('MCFC', **kwargs)
        self.mass_input_size = in_features
        self.aux_input_size = 1
        self.hidden_size = out_features
        self.normaliser = nn.Softmax(dim=-1)

        self.out_gate = Gate(self.hidden_size, self.aux_input_size)
        self.junction = get_redistribution("linear",
                                           num_states=self.mass_input_size,
                                           num_features=self.aux_input_size,
                                           num_out=self.hidden_size,
                                           normaliser=self.normaliser)

    @torch.no_grad()
    def reset_parameters(self):
        # perfect parameters for seed 45
        # self.out_gate.fc.weight[:] = torch.tensor([5., 5., -5.]).view(-1, 1)
        # self.out_gate.fc.bias[:] = torch.tensor([5., 5., -5.])
        # nn.init.constant_(self.junction.r, -5.)
        # self.junction.r[:11, -1] = 5.
        # self.junction.r[11:24, [0, -1]] = 0.
        # self.junction.r[24:36, [0, 1]] = 0.
        # self.junction.r[36:49, [1, -1]] = 0.
        # self.junction.r[49:, -1] = 5.
        self.out_gate.reset_parameters()
        # nn.init.constant_(self.out_gate.fc.bias, -3.)
        self.junction.reset_parameters()

    def log_gradients(self):
        for name, parameter in self.named_parameters():
            gradient, *_ = parameter.grad.data
            self.writer.add_summary(f'{name}/grad', gradient)
            self.writer.add_histogram(f'{name}/grad', gradient)

    def regualizer(self, merge_in=None):
        r1 = -torch.mean(self.junction.r ** 2)
        r2 = -torch.mean(self.out_gate.fc.weight ** 2)
        r3 = -torch.mean(self.out_gate.fc.bias ** 2)
        return super().regualizer({
            'W': r1 + r2 + r3
        })

    def forward(self, x):
        x_m, x_a = x, x.new_ones(1)
        j = self.junction(x_a)
        o = self.out_gate(x_a)

        m_in = torch.matmul(x_m.unsqueeze(-2), j).squeeze(-2)
        return o * m_in


class MulMCFC(ExtendedTorchModule):

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('MulMCFC', **kwargs)
        self.mcfc = MCFullyConnected(in_features, out_features + 1, **kwargs)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def reset_parameters(self):
        self.mcfc.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        log_sum = self.mcfc(torch.log(x))[:, :-1]
        return torch.exp(log_sum + self.bias)


class MultiplicativeLinear(ExtendedTorchModule):

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('MulLin', **kwargs)
        self.fc = nn.Linear(in_features, out_features)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def log_gradients(self):
        for name, parameter in self.named_parameters():
            gradient, *_ = parameter.grad.data
            self.writer.add_summary(f'{name}/grad', gradient)
            self.writer.add_histogram(f'{name}/grad', gradient)

    def forward(self, x):
        return self.fc(torch.log(x)).exp()


class PerfectProd(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return torch.prod(2 * x[:, :-1], dim=-1, keepdim=True)


class EnhancedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(torch.log(x)).exp()
