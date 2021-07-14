import argparse

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='PyTorch QAT')
    parser.add_argument('--tl','--transfer_learning',action='store_true',help='used to map weights correctly')
    parser.add_argument('--iter',default=300, type=int, help='no of iterations')
    parser.add_argument('--m','--model_name',default=None,help="Name of the model")
    parser.add_argument('--b', '--batch_size', default=128, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--optimizer', default='Adam', type=str,help='type of optimizer (default=Adam)')
    parser.add_argument( '--wd','--weight-decay', default=1e-5, type=float, help='weight decay (default: 1e-5)')
    parser.add_argument('--start_epoch','--s_ep', default=0, type=int, help='starting epoch')
    parser.add_argument('--num_epochs',default=30,type=int, help='no of epochs')
    parser.add_argument('--no_cuda', action='store_true',help='disables cuda training')
    parser.add_argument('--seed', type=int, default=12345,help='random seed for experiments. [default: 12345]')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lrdt', '--learning_rate_decay_interval', default=30, type=int, help='initial learning rate decay after n epochs')
    parser.add_argument('--od','--output_dir', default='/tmp/',help='output path')
    parser.add_argument('--en','--exp_name', default='pytorch_exp',help = 'experiment name to create output dir')
    parser.add_argument('--load_ckpt', default = None, help = "path to ckpt")
    parser.add_argument('--netqat',action='store_true',help = 'quantize model using custom layer')
    parser.add_argument('--partial_ckpt',action='store_true',help = 'load_partial checkpoint')
    parser.add_argument('--v','--verbose',action='store_true')
    parser.add_argument('--FP16',action='store_true',help='run TRT engine at FP16')
    parser.add_argument('--test_trt',action='store_true',help='gather metrics using trt')
    parser.add_argument('--INT8PTC',action='store_true',help='run TRT engine at INT8 with Post Training Cal')
    parser.add_argument('--INT8QAT',action='store_true',help='run TRT engine at INT8 with QAT')
    args = parser.parse_args()
    return args

