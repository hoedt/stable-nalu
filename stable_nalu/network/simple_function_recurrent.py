
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, GeneralizedCell

class SimpleFunctionRecurrentNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedCell.UNIT_NAMES | {'MulMCLSTM'}

    def __init__(self, unit_name, input_size=10, hidden_size=2, writer=None,
                 nac_mul='none', **kwargs):
        super().__init__('network', writer=writer, **kwargs)

        self.unit_name = unit_name
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Since for the 'mul' problem, the zero_state should be 1, and for the
        # 'add' problem it should be 0. The zero_states are allowed to be
        # # optimized.
        if unit_name == 'LSTM':
            self.zero_state = torch.nn.ParameterDict({
                'h_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size)),
                'c_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size))
            })
        elif unit_name in {'MCLSTM', 'MulMCLSTM'}:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        else:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        self.recurent_cell = GeneralizedCell(input_size, self.hidden_size,
                                             'MCLSTM' if unit_name == 'MulMCLSTM' else unit_name,
                                             writer=self.writer,
                                             name='recurrent_layer',
                                             **kwargs)

        if nac_mul == 'mnac':
            out_layer = unit_name[0:-3] + 'MNAC'
        elif unit_name in {'GRU', 'LSTM', 'MCLSTM', 'RNN-tanh', 'RNN-ReLU'}:
            out_layer = 'linear'
        elif unit_name == 'MulMCLSTM':
            out_layer = 'MulMCFC'
        else:
            out_layer = unit_name

        # TODO: check effect of MNAC parameter!
        self.output_layer = GeneralizedLayer(self.hidden_size, 1,
                                             out_layer,
                                             writer=self.writer,
                                             name='output_layer',
                                             **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            for zero_state in self.zero_state.values():
                torch.nn.init.zeros_(zero_state)
        else:
            torch.nn.init.zeros_(self.zero_state)

        self.recurent_cell.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, dims]
        """
        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = tuple(zero_state.repeat(x.size(0), 1) for zero_state in self.zero_state.values())
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)

        if self.unit_name.endswith('MCLSTM'):
            auxiliaries = x.new_ones(*x.shape[:2], 1)
            auxiliaries[:, -1] = -1

        for t in range(x.size(1)):
            x_t = x[:, t]

            if self.unit_name.endswith('MCLSTM'):
                h_t = self.recurent_cell(x_t, auxiliaries[:, t], h_tm1)
            else:
                h_t = self.recurent_cell(x_t, h_tm1)
            h_tm1 = h_t[1] if self.unit_name.endswith('MCLSTM') else h_t

        # Grap the final hidden output and use as the output from the recurrent layer
        z_1 = h_t[0] if self.unit_name.endswith('LSTM') else h_t
        z_2 = self.output_layer(z_1)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
