import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)

def add_logging_argument(parser):
    parser.add_argument('--wandb_prj', type=str, default='topmost')


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name, currently support datasets are: \
                            20NG, ACL, Amazon_Review, ECNews, IMDB, NeurIPS, \
                            NYT, Rakuten_Amazon, Wikitext-103')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_groups', type=int, default=10)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=False, help='Enable use_pretrainWE mode')
    parser.add_argument('--weight_ECR', type=float, default=250.)
    parser.add_argument('--weight_XGR', type=float, default=250.)
    parser.add_argument('--alpha_ECR', type=float, default=20.)
    parser.add_argument('--alpha_XGR', type=float, default=5.)
    parser.add_argument('--weight_DCR', type=float, default=250.)
    parser.add_argument('--alpha_DCR', type=float, default=20.)
    parser.add_argument('--weight_TCR', type=float, default=250.)
    parser.add_argument('--alpha_TCR', type=float, default=20.)
    parser.add_argument('--weight_MMI', type=float, default=100.)
    parser.add_argument('--weight_TPD', type=float, default=20.)
    parser.add_argument('--alpha_TPD', type=float, default=20.)
    parser.add_argument('--gating_func', type=str, default='dot_bias')
    parser.add_argument('--weight_global_expert', type=float, default=250.)
    parser.add_argument('--weight_local_expert', type=float, default=250.)
    parser.add_argument('--k', help='top k expert', type=int, default=1)
    parser.add_argument('--beta_temp', type=float, default=0.2)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--rho', type=float, default=0.005,
                        help='rho')   
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma') 
    parser.add_argument('--lmbda', type=float, default=0.6,
                        help='lmbda')               
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')
    # parser.add_argument('--MOO_algo', type=str, default='IMTL',
    #                     help='Multi-Objective Optimization Algorithm')

def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
