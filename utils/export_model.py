import argparse
import sys

import torch
from neural_pipeline import Predictor, FileStructManager
from torch.jit import trace

from train_config import create_model, MyTrainConfig

dummy_input1 = torch.randn(1, 3, 224, 224)


def to_onnx(model, file_name: str):
    torch.onnx.export(model, dummy_input1, file_name + ".onnx", output_names=['output'], input_names=['input'])


def to_libtorch(model, file_name: str):
    traced_script_module = trace(model, (dummy_input1,))
    traced_script_module.save(file_name + ".pt")


class Model(torch.nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.activation = torch.nn.Sigmoid()

    def forward(self, data):
        return self.activation(self.model(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('-e', '--exp_name', type=str, help='Experiment name', required=False)
    parser.add_argument('-t', '--target', type=str, help='Target format', choices=['onnx', 'libtorch'], required=True)
    parser.add_argument('-o', '--out', type=str, help='Output file path', required=True)

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    model = create_model().eval()

    if args.exp_name is not None:
        file_struct_manager = FileStructManager(base_dir=MyTrainConfig.experiment_dir, is_continue=True)
        predictor = Predictor(model, file_struct_manager)
    model = Model(model)

    if args.target == 'onnx':
        to_onnx(model, args.out)
    elif args.target == 'libtorch':
        to_libtorch(model, args.out)
