import argparse

def get_args():
    parser = argparse.ArgumentParser('Arguments for constrastive learning.')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--device', type=str, default='cpu')
    

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model_arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--data_folder', type=str, default='../data', help='path to custom dataset')

    # checkpoint to evaluate
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # SAM parameters
    parser.add_argument('--sam', default=False, action='store_true',
                        help='whether to use SAM optimizer')
    parser.add_argument('--adaptive', default=True, action='store_true',
                        help='whether to use the adaptive version of SAM')
    parser.add_argument('--rho', type=float, default=2.0,
                        help='rho parameter for loss (A)SAM')
    
    # AT training
    parser.add_argument('--at', default=False, action='store_true',
                        help='whether to use AT for linear')
    
    parser.add_argument('--run_unseen_attacks', default=False, type=bool)
    

    parser.add_argument('--seed', type=int, default=0x6e676f63746e71,
                        help='random seed')

    args = parser.parse_args()

    # LR scheduling
    args.eta_min = 1e-8

    return args
