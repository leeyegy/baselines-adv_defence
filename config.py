import argparse
parser = argparse.ArgumentParser(description='Train MNIST')
parser.add_argument('--seed', default=0, type=int)
# attack
parser.add_argument("--attack_method", default="PGD", type=str,
                    choices=['FGSM', 'PGD', 'Momentum', 'STA', "DeepFool", "CW","NONE"])
parser.add_argument('--epsilon', type=float, default=0.00784, help='if adopt pixelcnn, epsilon should use int type')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

# net
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--num_classes', default=10, type=int)


# defence
parser.add_argument('--defence_method', default="FeatureSqueezing", type=str, choices=['FeatureSqueezing',"TotalVarMin","SpatialSmoothing","JPEGCompression"])
args = parser.parse_args()
