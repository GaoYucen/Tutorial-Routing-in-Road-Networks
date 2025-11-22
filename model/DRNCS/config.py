# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description="Pytorch implementation of Road-SPDE")

# Model
parser.add_argument('--nof_epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--dimen', default=500, type=int, help='Dimension of embedding')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--nof_samples', default=1000000, type=int, help='Number of samples')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
parser.add_argument('--input_dimension', default=2, type=int, help='input dimension')
parser.add_argument('--embedding_dimension', default=128, type=int, help='embedding dimension')

# Data
parser.add_argument('--num', default=3, type=int, help='Data number')
parser.add_argument('--reverse_num', default=1000, type=int, help='Reverse Data number')
parser.add_argument('--dis_node_num', default=1000, type=int, help='Distance Data number')
parser.add_argument('--output_dimen', default=256, type=int, help='Output dimension')
parser.add_argument('--tildeL1_ratio', default=0.99, type=int, help='tildeL1 ratio')
parser.add_argument('--g_training_epochs', default=10, type=int, help='epochs of global training')


# NO Direct
parser.add_argument('--nodirect', default=False, type=bool, help='Whether to use no direct graph')
parser.add_argument('--sim', default=False, type=bool, help='Sim or not')
parser.add_argument('--city', default='no', type=str, help='city name')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed