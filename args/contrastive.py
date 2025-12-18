import argparse

def get_args():
    parser = argparse.ArgumentParser('Arguments for constrastive learning.')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--device', type=str, default='cpu')
    

    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model_arch', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--data_folder', type=str, default='../data', help='path to custom dataset')
    parser.add_argument('--cj_str', type=float, default=0.5,
                        help='color jitter strength')

    # reload checkpoint
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # attack parameters
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='perturbation budget')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='attack step size')
    parser.add_argument('--nb_iter', type=int, default=7,
                        help='number of attack iterations; 0 for no attack and 1 for FGSM')
    parser.add_argument('--rosa', default=False, action='store_true',
                        help='whether to use RoSA with discriminator')

    # SAM parameters
    parser.add_argument('--sam', default=False, action='store_true',
                        help='whether to use SAM optimizer')
    parser.add_argument('--adaptive', default=False, action='store_true',
                        help='whether to use the adaptive version of SAM')
    parser.add_argument('--rho', type=float, default=2.0,
                        help='rho parameter for loss (A)SAM')
    parser.add_argument('--benign', default=False, action='store_true',
                        help='use SAM for only benign')
    
    parser.add_argument('--benign_w', type=float, default=1.0,
                        help='weight for benign loss')
    

    # Discriminator parameters
    parser.add_argument('--discr', default=False, action='store_true',
                        help='whether to use Discriminator')
    parser.add_argument('--d_num_layer', type=int, default=0, choices=[0,1],
                        help='number of layers for D')
    parser.add_argument('--learning_rate_d', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--d_w', type=float, default=1.0,
                        help='weight for D_loss')
    parser.add_argument('--d_step', type=int, default=3, choices=[3,4],
                        help='train with 3-step or 4-step')
    parser.add_argument('--use_d_gen', default=False, action='store_true',
                        help='use D when gen adv of 3 step?')
    parser.add_argument('--start_from', type=int, default=0,
                        help='use D from epoch')
    parser.add_argument('--d_reduction', type=str, default='mean', choices=['sum', 'mean'],
                        help='reduction for BCE loss')
    parser.add_argument('--d_optim', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='D optimizer')

    # Strategy for training SSL
    parser.add_argument('--strategy', type=str, default='base', 
                        choices=['base', 'adv', 'advcl_baseline'])
    parser.add_argument('--method_gen', type=str, default='rocl',
                         choices=['rocl', 'ae4cl', 'aclds'])
    parser.add_argument('--method_loss', type=str, default='rocl_new',
                         choices=['rocl_new', 'ae4cl', 'rocl', 'aclds'])
    

    # experiment management
    parser.add_argument('--exp_id', type=str, default='-1',
                        help='Experiment ID, -1 for omission')
    parser.add_argument('--run_eval', default=False, action='store_true',
                        help='run linear eval right after training?')
    parser.add_argument('--run_eval_bn', default=False, action='store_true',
                        help='run linear eval with correct bn right after training?')
    
    parser.add_argument('--seed', type=int, default=0x6e676f63746e71,
                        help='random seed')

    parser.add_argument('--learning_rate_linear', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--run_unseen_attacks', default=False, type=bool)
    
    parser.add_argument('--original_at', type=str, default='default',
                        help='positive as original')
    
    parser.add_argument('--benign_type', type=str, default='simclr', choices=['advcl', 'simclr'],
                        help='benign type for advcl_ours')

    parser.add_argument('--scheduler', type=str, default='ours', choices=['torch', 'ours'],
                        help='type for scheduler')
    
    parser.add_argument('--criterion_type', type=str, default='infonce', choices=['infonce', 'supcon', 'nt_xent'],
                        help='benign type for criterin')


    
    args = parser.parse_args()

    get_save_folder(args)
    
    # LR scheduling
    args.eta_min = 1e-8
    args.warm_epochs = 10

    return args

def get_save_folder(args):
    args.save_folder = './aistats25/checkpoints/{}/{}/{}_{}_{}_lr_{}_bsz_{}_epoch_{}'.format(
        args.dataset, args.seed, args.exp_id, args.model_arch, args.optim, args.learning_rate, args.batch_size, args.epochs
    )
    if args.sam:
        if args.adaptive:
            args.save_folder += f'_asam_{args.rho}'
        else:
            args.save_folder += f'_sam_{args.rho}'

    args.save_folder += f'_stra_{args.strategy}'
    if args.strategy in ['adv', 'advcl_baseline']:
        args.save_folder += f'_benign_{args.benign}'
        args.save_folder += f'_benign_w_{args.benign_w}'
        args.save_folder += f'_method_gen_{args.method_gen}'
        args.save_folder += f'_method_loss_{args.method_loss}'
        # args.save_folder += '_ep_{}_alpha_{}_k_{}'.format(args.epsilon, args.alpha, args.nb_iter)
    if args.strategy == 'advcl_baseline':
        args.save_folder += f'_{args.benign_type}'

    if args.discr:
        args.save_folder += f'_discr_{args.d_num_layer}_layer_{args.d_optim}_lrd_{args.learning_rate_d}'
        args.save_folder += f'_dw_{args.d_w}_dstep_{args.d_step}_use_d_gen_{args.use_d_gen}'
        args.save_folder += f'_from_{args.start_from}'
        
    