import argparse
import datetime
from utils import logging
from models import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic_nonlinear',
                        help='dataset to use')
    parser.add_argument('--fid', type=str, default=1,
                        help='id for run')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='path to data files')
    parser.add_argument('--log_dir', type=str, default='test/',
                        help='directory for logging')
    parser.add_argument('--img_size_w', type=int, default=28,
                        help='width of input images')
    parser.add_argument('--img_size_h', type=int, default=28,
                        help='height of input images')
    parser.add_argument('--n_channels', type=int, default=1,
                        help='number of channels of input images')
    parser.add_argument('--n_input', type=int, default=2,
                        help='dim of input')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--architecture', type=str, default='nonlinear',
                        help='conv, nonlinear or linear')
    # batch norm
    parser.add_argument('--bn', dest='bn', action='store_true')
    parser.add_argument('--no-bn', dest='bn', action='store_false')
    parser.set_defaults(bn=False)
    parser.add_argument('--batch_size', type=int, default=120,
                        help='minibatch size')
    parser.add_argument('--eval_batch_size', type=int, default=200,
                        help='minibatch size evaluation')
    parser.add_argument('--num_epochs_class', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate_class', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9999,
                        help='decay rate for learning rate')
    parser.add_argument('--h1', type=int, default=500,
                        help='dim of hidden layer 1')
    parser.add_argument('--h2', type=int, default=500,
                        help='dim of  hidden layer 2')
    parser.add_argument('--img_data', dest='img_data', action='store_true')
    parser.add_argument('--no-img_data', dest='img_data', action='store_false')
    parser.set_defaults(img_data=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--do-not-normalize', dest='normalize',
                        action='store_false')
    parser.set_defaults(normalize=False)
    parser.add_argument('--classifier', type=str, default='counterfactual',
                        help='counterfactual, max_loss, or standard')
    parser.add_argument('--weight_countfact_loss', type=float, default=1,
                        help='weight counterfactual loss')
    parser.add_argument('--cfl_annealing', dest='cfl_annealing',
                        action='store_true')
    parser.add_argument('--no-cfl_annealing', dest='cfl_annealing',
                        action='store_false')
    parser.set_defaults(cfl_annealing=False)
    parser.add_argument('--cfl_rate_rise_factor', type=float, default=0.01,
                        help='recon loss weight is increasd by this much \
                        every save_every steps')
    parser.add_argument('--cfl_rate_rise_time', type=int, default=30000,
                        help='iterations before increasing cf loss term')
    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.add_argument('--no-regression', dest='regression',
                        action='store_false')
    parser.set_defaults(regression=False)
    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                        help='lambda_reg')
    parser.add_argument('--save_img_sums', dest='save_img_sums',
                        action='store_true')
    parser.add_argument('--do-not-save_img_sums', dest='save_img_sums',
                        action='store_false')
    parser.set_defaults(save_img_sums=False)
    parser.add_argument('--two_layer_CNN', dest='two_layer_CNN',
                        action='store_true')
    parser.add_argument('--do-not-use-two_layer_CNN', dest='two_layer_CNN',
                        action='store_false')
    parser.set_defaults(two_layer_CNN=True)
    args = parser.parse_args()

    args.time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
    modelparam = logging.make_modelparam_string_class(args)
    train.train(args, modelparam)


if __name__ == '__main__':
    main()
