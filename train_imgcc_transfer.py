from utils.logger import setlogger
import logging
# from utils.train_utils import train_utils
from utils.train_utils_multicon_cc import train_utils

import os
from datetime import datetime
import argparse

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters

    # parser.add_argument('--model_name', type=str, default='lenet', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='alexnet', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='cnn', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='vgg16', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='inception', help='the name of the model')
    parser.add_argument('--model_name', type=str, default='resnet18', help='the name of the model')

    # 单通道测试
    # parser.add_argument('--model_name', type=str, default='lenet_2d1c', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='alexnet_2d1c', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='cnn_2d1c', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='vgg16_2d1c', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='inception_2d1c', help='the name of the model')
    # parser.add_argument('--model_name', type=str, default='resnet18_2d1c', help='the name of the model')

    # 增加一个图片的通道类型
    # parser.add_argument('--img_channel', type=int, default=1, help='input images channel, 3 or 1')
    parser.add_argument('--img_channel', type=int, default=3, help='input images channel, 3 or 1')



    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')

    # parser.add_argument('--data_dir', type=str, default="E:\PycharmProjects\data\CWRU",
    #                     help='the directory of the data') # 在笔记本中

    # parser.add_argument('--data_dir', type=str, default="/Users/liufei/PycharmProjects/CWRU",
    #                     help='the directory of the data') # macos下
    parser.add_argument('--data_dir', type=str, default="F:\liufeidata\datasets\CWRU",
                        help='the directory of the data') # 在工作站下

    # parser.add_argument('--data_dir', type=str, default="/content/CWRU",
    #                     help='the directory of the data')   # 在google的colab平台上，将默认数据集路径改掉就好了

    # 增加一个迁移学习的
    parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    # parser.add_argument('--transfer_task', type=list, default=[[2], [3]], help='transfer learning tasks')

    # parser.add_argument('--adabn', type=bool, default=False, help='whether using adabn')


    # 是否使用基于模型的迁移学习（利用目标域数据微调微调）
    # parser.add_argument('--finetune', type=bool, default=True, help='whether using target fine-tune')
    parser.add_argument('--finetune', type=bool, default=False, help='whether using target fine-tune')
    parser.add_argument('--testmodel', type=str, default='best', help='choose which pretrain model in target, best or last')


    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')

    parser.add_argument('--datasets_type', type=str, default='CWRU_STFT',
                        help='选择数据集的具体类型')

    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')



    # 感兴趣的
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    # parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')






    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    # parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    # parser.add_argument('--batch_size', type=int, default=10, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')





    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer') #优化器的选择

    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')









    # save, load and display information
    # parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=2, help='max number of epoch')
    # 新增的对fine tune 训练的循环次数
    parser.add_argument('--target_train_epoch', type=int, default=1, help='max number of epoch')

    # parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = args.model_name+'_'+args.datasets_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    #手动运行测试集
    # sub_dir = args.model_name + '_' + args.datasets_type + '_' + '0714-215627'

    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()

    trainer.train()
    trainer.transfer_condition()






