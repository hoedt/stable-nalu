import ast
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionRecurrentNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--hidden-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the hidden size')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--input-size',
                    action='store',
                    default=10,
                    type=int,
                    help='Specify the input size')
parser.add_argument('--seq-length',
                    action='store',
                    default=10,
                    type=int,
                    help='Specify the sequence length')
parser.add_argument('--num-subsets',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the number of subsets to use')
parser.add_argument('--subset-ratio',
                    action='store',
                    default=0.25,
                    type=float,
                    help='Specify the subset-size as a fraction of the input-size')
parser.add_argument('--overlap-ratio',
                    action='store',
                    default=0.5,
                    type=float,
                    help='Specify the overlap-size as a fraction of the input-size')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')
parser.add_argument('--interpolation-range',
                    action='store',
                    default=[1, 2],
                    type=ast.literal_eval,
                    help='Specify the interpolation range that is sampled uniformly from')
parser.add_argument('--extrapolation-range',
                    action='store',
                    default=[2, 6],
                    type=ast.literal_eval,
                    help='Specify the extrapolation range that is sampled uniformly from')

parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')

parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')

parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--regualizer-scaling',
                    action='store',
                    default='linear',
                    choices=['exp', 'linear'],
                    type=str,
                    help='Use an expoentational scaling from 0 to 1, or a linear scaling.')
parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=1000000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=2000000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer.')
parser.add_argument('--l2-out',
                    action='store',
                    default=0.,
                    type=float,
                    help='use L2 regularisation on the output layer')
args = parser.parse_args()


setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print(f'running')
print(f'  - layer_type: {args.layer_type}')
print(f'  - hidden_size: {args.hidden_size}')
print(f'  - operation: {args.operation}')
print(f'  - regualizer: {args.regualizer}')
print(f'  - regualizer_z: {args.regualizer_z}')
print(f'  - regualizer_oob: {args.regualizer_oob}')
print(f'  - l2_out: {args.l2_out}')
print(f'  -')
print(f'  - max_iterations: {args.max_iterations}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  - learning_rate: {args.learning_rate}')
print(f'  -')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  -')
print(f'  - interpolation_range: {args.interpolation_range}')
print(f'  - extrapolation_range: {args.extrapolation_range}')
print(f'  - input_size: {args.input_size}')
print(f'  - seq_length: {args.seq_length}')
print(f'  - num_subsets: {args.num_subsets}')
print(f'  - subset_ratio: {args.subset_ratio}')
print(f'  - overlap_ratio: {args.overlap_ratio}')
print(f'  - simple: {args.simple}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')

# Prepear logging
# results_writer = stable_nalu.writer.ResultsWriter('simple_function_recurrent')
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-nac-" if args.nac_mul != "none" else ""}'
    f'{"n" if args.nac_mul == "normal" else ""}'
    f'{"s" if args.nac_mul == "safe" else ""}'
    f'{"s" if args.nac_mul == "max-safe" else ""}'
    f'{"t" if args.nac_mul == "trig" else ""}'
    f'{"m" if args.nac_mul == "mnac" else ""}'
    f'_op-{args.operation.lower()}'
    f'_rs-{args.regualizer_scaling}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}-{args.l2_out}'
    f'_i-{args.interpolation_range[0]}-{args.interpolation_range[1]}'
    f'_e-{args.extrapolation_range[0]}-{args.extrapolation_range[1]}'
    f'_z-{"simple" if args.simple else f"{args.seq_length}-{args.input_size}-{args.subset_ratio}-{args.overlap_ratio}"}'
    f'_h{args.hidden_size}'
    f'_z{args.num_subsets}'
    f'_lr-{args.learning_rate}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    '_fix-junc',
    remove_existing_data=args.remove_existing_data
)

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionRecurrentDataset(
    operation=args.operation,
    input_size=args.input_size,
    subset_ratio=args.subset_ratio,
    overlap_ratio=args.overlap_ratio,
    num_subsets=args.num_subsets,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed
)
print(f'  -')
print(f'  - dataset: {dataset.print_operation()}')

dataset_train = iter(dataset.fork(seq_length=args.seq_length, input_range=args.interpolation_range).dataloader(batch_size=args.batch_size))
dataset_valid_interpolation = iter(dataset.fork(seq_length=args.seq_length, input_range=args.interpolation_range, seed=43953907).dataloader(batch_size=1000))
dataset_valid_extrapolation = iter(dataset.fork(seq_length=2 * args.seq_length, input_range=args.interpolation_range, seed=2389231).dataloader(batch_size=1000))
dataset_test_extrapolation = iter(dataset.fork(seq_length=10 * args.seq_length, input_range=args.interpolation_range, seed=8689336).dataloader(batch_size=1000))
dataset_test_extrapolation_range = iter(dataset.fork(seq_length=args.seq_length, input_range=args.extrapolation_range, seed=23946).dataloader(batch_size=1000))

# setup model
model = stable_nalu.network.SimpleFunctionRecurrentNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    hidden_size=args.hidden_size,
    writer=summary_writer.every(1000) if args.verbose else None,
    first_layer=None,
    nac_oob='clip',
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=0.,
    nac_mul=args.nac_mul,
    nalu_bias=False,
    nalu_two_nac=False,
    nalu_two_gate=False,
    nalu_mul='normal',
    nalu_gate='normal',
    junct='linear'
)
if args.cuda:
    model.cuda()
model.reset_parameters()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, min_lr=1e-5, verbose=True)

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging():
        x, t = next(dataloader)
        return criterion(model(x), t)

# Train model
for epoch_i, (x_train, t_train) in zip(range(args.max_iterations + 1), dataset_train):
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % 1000 == 0:
        interpolation_error = test_model(dataset_valid_interpolation)
        extrapolation_error = test_model(dataset_valid_extrapolation)
        seq_extrapolation_error = test_model(dataset_test_extrapolation)
        range_extrapolation_error = test_model(dataset_test_extrapolation_range)

        summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
        summary_writer.add_scalar('metric/valid/extrapolation', extrapolation_error)
        summary_writer.add_scalar('metric/test/extrapolation', seq_extrapolation_error)
        summary_writer.add_scalar('metric/test/range_extrapolation', range_extrapolation_error)

    # forward
    y_train = model(x_train)
    regualizers = model.regualizer()

    if (args.regualizer_scaling == 'linear'):
        r_w_scale = max(0, min(1, (
            (epoch_i - args.regualizer_scaling_start) /
            (args.regualizer_scaling_end - args.regualizer_scaling_start)
        )))
    elif (args.regualizer_scaling == 'exp'):
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    loss_train_criterion = criterion(y_train, t_train)
    loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + regualizers['g'] + args.regualizer_z * regualizers['z'] + args.regualizer_oob * regualizers['W-OOB']
    if args.l2_out > 0:
        loss_train_regualizer += args.l2_out * torch.mean(model.output_layer.layer.W ** 2) / 2
    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
    summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
    summary_writer.add_scalar('loss/train/total', loss_train)
    if epoch_i % 1000 == 0:
        # lr_schedule.step(loss_train)
        print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % 1000 == 0:
        model.log_gradients()

# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation)
loss_valid_seq = test_model(dataset_valid_extrapolation)
loss_test_seq = test_model(dataset_test_extrapolation)
loss_test_range = test_model(dataset_test_extrapolation_range)

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid_inter: {loss_valid_inter}')
print(f'  - loss_valid_seq: {loss_valid_seq}')
print(f'  - loss_test_seq: {loss_test_seq}')
print(f'  - loss_test_range: {loss_test_range}')

# Use saved weights to visualize the intermediate values.
stable_nalu.writer.save_model(summary_writer.name, model)
