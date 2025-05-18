import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='DTrOCR w. CTC + SAM',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out-dir', type=str, default='./output', help='output directory')
    parser.add_argument('--train-bs', default=8, type=int, help='train batch size')
    parser.add_argument('--val-bs', default=1, type=int, help='validation batch size')
    parser.add_argument('--eval-bs', default=1, type=int, help='eval batch size')
    parser.add_argument('--epochs', default=5, type=int, help='nb of epochs')
    parser.add_argument('--num-workers', default=0, type=int, help='nb of workers')
    parser.add_argument('--eval-iter', default=10000, type=int, help='nb of iterations to run evaluation')
    parser.add_argument('--total-iter', default=100000, type=int, help='nb of total iterations for training')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='nb of iterations for warm-up')
    parser.add_argument('--print-iter', default=100, type=int, help='nb of total iterations to print information')
    parser.add_argument('--max-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=5e-1, type=float, help='weight decay')
    parser.add_argument('--use-wandb', action='store_true', default=False, help = 'whether use wandb, otherwise use tensorboard')
    parser.add_argument('--exp-name',type=str, default='DTrOCR', help='experimental name (save dir will be out_dir + exp_name)')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')

    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+', help='image size')

    parser.add_argument('--ema-decay', default=0.9999, type=float, help='Exponential Moving Average (EMA) decay')
    parser.add_argument('--alpha', default=0, type=float, help='kld loss ratio')

    parser.add_argument('--load-model', type=str, default='best_CER', help='Load model eg. for validation')
    parser.add_argument('--dec-model', type=str, default='gpt2', choices=['gpt2', 'gpt2-it'], help='Choose decoder model based on gpt2 architecture')
    parser.add_argument('--dataset-name', type=str, default='vpippi/lam-dataset', help='Select dataset name')
    parser.add_argument('--plot-eval', action='store_true', default=False, help = 'whether to plot graph at the eval stage')
    
    return parser.parse_args()
