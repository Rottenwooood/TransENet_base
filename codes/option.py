import argparse
import template

parser = argparse.ArgumentParser(description='Super-resolution')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Enables debug mode')

# hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# data specifications
parser.add_argument('--dataset', type=str, default='UCMerced',
                    help='train dataset name')
parser.add_argument('--dir_data', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--dir_out', type=str, default='./output',
                    help='output directory')
parser.add_argument('--data_train', type=str, default='/root/autodl-tmp/TransENet/datasets/UCMerced-train/UCMerced-dataset/train',
                    help='training dataset directory')
parser.add_argument('--data_val', type=str, default='/root/autodl-tmp/TransENet/datasets/UCMerced-train/UCMerced-dataset/val',
                    help='validation dataset directory')
parser.add_argument('--data_test', type=str, default='.',
                    help='test dataset name')
parser.add_argument('--image_size', type=int, default=256,
                    help='train/test reference image size')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size for training')
parser.add_argument('--cubic_input', action='store_true', default=False,
                    help='LR images are firstly upsample by cubic interpolation')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension: '
                         'sep_reset - first convert img to .npy and read .npy; '
                         'sep - read .npy from disk; '
                         'img - read image from disk; '
                         'ram - load image into RAM memory; '
                         'pt - load from .pt file')

parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true', default=False,
                    help='enable memory-efficient forward')
parser.add_argument('--test_y', action='store_true', default=False,
                    help='test on Y channel')
parser.add_argument('--test_patch', action='store_true', default=False,
                    help='test on patches rather than the whole image')
parser.add_argument('--test_block', action='store_true', default=False,
                    help='test by blcok-by-block')

# model specifications
parser.add_argument('--model', default='BASIC',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--n_basic_modules', type=int, default=10,
                    help='number of basic modules')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# training specifications
parser.add_argument('--reset', action='store_true', default=False,
                    help='reset the training')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true', default=False,
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', default=False,
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--test_metric', type=str, default='psnr',
                    help='for best model selection in test phase (psnr, ssim)')

# optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=400,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'ADAMW', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | ADAMW | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_false', default=True,
                    help='print model')
parser.add_argument('--save_models', action='store_true', default=False,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=False,
                    help='save output results')

# Option for TransENet
parser.add_argument('--back_projection_iters', type=int, default=10,
                    help='back projection iterations')
parser.add_argument('--en_depth', type=int, default=8,
                    help='the depth of encoder')
parser.add_argument('--de_depth', type=int, default=1,
                    help='the depth of decoder')

# Option for SymUNet
parser.add_argument('--symunet_width', type=int, default=64,
                    help='base number of channels for SymUNet')
parser.add_argument('--symunet_middle_blk_num', type=int, default=1,
                    help='number of middle blocks in SymUNet')
parser.add_argument('--symunet_enc_blk_nums', type=str, default='2,2,2',
                    help='number of encoder blocks for each stage (comma-separated)')
parser.add_argument('--symunet_dec_blk_nums', type=str, default='2,2,2',
                    help='number of decoder blocks for each stage (comma-separated)')
parser.add_argument('--symunet_ffn_expansion_factor', type=float, default=2.66,
                    help='FFN expansion factor for SymUNet')
parser.add_argument('--symunet_bias', action='store_true', default=False,
                    help='use bias in SymUNet')
parser.add_argument('--symunet_layer_norm_type', type=str, default='WithBias',
                    choices=('WithBias', 'BiasFree'),
                    help='Layer normalization type for SymUNet')
parser.add_argument('--symunet_restormer_heads', type=str, default='1,2,4',
                    help='number of attention heads for each encoder/decoder stage (comma-separated)')
parser.add_argument('--symunet_restormer_middle_heads', type=int, default=8,
                    help='number of attention heads for middle blocks in SymUNet')

# Option for SymUNet Pre-train (预上采样版本)
parser.add_argument('--symunet_pretrain_width', type=int, default=64,
                    help='base number of channels for SymUNet-Pretrain')
parser.add_argument('--symunet_pretrain_middle_blk_num', type=int, default=1,
                    help='number of middle blocks in SymUNet-Pretrain')
parser.add_argument('--symunet_pretrain_enc_blk_nums', type=str, default='2,2,2',
                    help='number of encoder blocks for each stage in SymUNet-Pretrain (comma-separated)')
parser.add_argument('--symunet_pretrain_dec_blk_nums', type=str, default='2,2,2',
                    help='number of decoder blocks for each stage in SymUNet-Pretrain (comma-separated)')
parser.add_argument('--symunet_pretrain_ffn_expansion_factor', type=float, default=2.66,
                    help='FFN expansion factor for SymUNet-Pretrain')
parser.add_argument('--symunet_pretrain_bias', action='store_true', default=False,
                    help='use bias in SymUNet-Pretrain')
parser.add_argument('--symunet_pretrain_layer_norm_type', type=str, default='WithBias',
                    choices=('WithBias', 'BiasFree'),
                    help='Layer normalization type for SymUNet-Pretrain')
parser.add_argument('--symunet_pretrain_restormer_heads', type=str, default='1,2,4',
                    help='number of attention heads for each encoder/decoder stage in SymUNet-Pretrain (comma-separated)')
parser.add_argument('--symunet_pretrain_restormer_middle_heads', type=int, default=8,
                    help='number of attention heads for middle blocks in SymUNet-Pretrain')

# Option for SymUNet Post-train (后上采样版本)
parser.add_argument('--symunet_posttrain_width', type=int, default=64,
                    help='base number of channels for SymUNet-Posttrain')
parser.add_argument('--symunet_posttrain_middle_blk_num', type=int, default=1,
                    help='number of middle blocks in SymUNet-Posttrain')
parser.add_argument('--symunet_posttrain_enc_blk_nums', type=str, default='2,2,2',
                    help='number of encoder blocks for each stage in SymUNet-Posttrain (comma-separated)')
parser.add_argument('--symunet_posttrain_dec_blk_nums', type=str, default='2,2,2',
                    help='number of decoder blocks for each stage in SymUNet-Posttrain (comma-separated)')
parser.add_argument('--symunet_posttrain_ffn_expansion_factor', type=float, default=2.66,
                    help='FFN expansion factor for SymUNet-Posttrain')
parser.add_argument('--symunet_posttrain_bias', action='store_true', default=False,
                    help='use bias in SymUNet-Posttrain')
parser.add_argument('--symunet_posttrain_layer_norm_type', type=str, default='WithBias',
                    choices=('WithBias', 'BiasFree'),
                    help='Layer normalization type for SymUNet-Posttrain')
parser.add_argument('--symunet_posttrain_restormer_heads', type=str, default='1,2,4',
                    help='number of attention heads for each encoder/decoder stage in SymUNet-Posttrain (comma-separated)')
parser.add_argument('--symunet_posttrain_restormer_middle_heads', type=int, default=8,
                    help='number of attention heads for middle blocks in SymUNet-Posttrain')

# WandB monitoring
parser.add_argument('--use_wandb', action='store_true', default=False,
                    help='use wandb for experiment tracking')
parser.add_argument('--wandb_project', type=str, default='SymUNet-SR',
                    help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='wandb entity name')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='wandb experiment name')

# Enhanced training options
parser.add_argument('--scheduler', default='step',
                    choices=('step', 'cosine'),
                    help='learning rate scheduler (step | cosine)')
parser.add_argument('--cosine_t_max', type=int, default=300,
                    help='maximum steps for cosine annealing')
parser.add_argument('--cosine_eta_min', type=float, default=5e-5,
                    help='minimum learning rate for cosine annealing')
parser.add_argument('--save_every_n_steps', type=int, default=50,
                    help='save checkpoint every n steps')

args = parser.parse_args()
args.scale = list(map(lambda x: int(x), args.scale.split('+')))

# Parse SymUNet parameters
args.symunet_enc_blk_nums = list(map(lambda x: int(x), args.symunet_enc_blk_nums.split(',')))
args.symunet_dec_blk_nums = list(map(lambda x: int(x), args.symunet_dec_blk_nums.split(',')))
args.symunet_restormer_heads = list(map(lambda x: int(x), args.symunet_restormer_heads.split(',')))

# Parse SymUNet-Pretrain parameters
args.symunet_pretrain_enc_blk_nums = list(map(lambda x: int(x), args.symunet_pretrain_enc_blk_nums.split(',')))
args.symunet_pretrain_dec_blk_nums = list(map(lambda x: int(x), args.symunet_pretrain_dec_blk_nums.split(',')))
args.symunet_pretrain_restormer_heads = list(map(lambda x: int(x), args.symunet_pretrain_restormer_heads.split(',')))

# Parse SymUNet-Posttrain parameters
args.symunet_posttrain_enc_blk_nums = list(map(lambda x: int(x), args.symunet_posttrain_enc_blk_nums.split(',')))
args.symunet_posttrain_dec_blk_nums = list(map(lambda x: int(x), args.symunet_posttrain_dec_blk_nums.split(',')))
args.symunet_posttrain_restormer_heads = list(map(lambda x: int(x), args.symunet_posttrain_restormer_heads.split(',')))

template.set_template(args)