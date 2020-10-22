import logging
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import models
import os
from datetime import datetime
import argparse
from utils.logger import setlogger
import numpy as np


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
from matplotlib import cm

# Scale and visualize the embedding vectors
# # def plot_embedding(X, title=None):
# #     x_min, x_max = np.min(X, 0), np.max(X, 0)
# #     X = (X - x_min) / (x_max - x_min)
# #
# #     plt.figure()
# #     ax = plt.subplot(111)
# #     for i in range(X.shape[0]):
# #         plt.text(X[i, 0], X[i, 1], str(y[i]),
# #                  color=plt.cm.Set1(y[i] / 10.),
# #                  fontdict={'weight': 'bold', 'size': 9})
# #
# #     if hasattr(offsetbox, 'AnnotationBbox'):
# #         # only print thumbnails with matplotlib > 1.0
# #         shown_images = np.array([[1., 1.]])  # just something big
# #         for i in range(X.shape[0]):
# #             dist = np.sum((X[i] - shown_images) ** 2, 1)
# #             if np.min(dist) < 4e-3:
# #                 # don't show points that are too close
# #                 continue
# #             shown_images = np.r_[shown_images, [X[i]]]
# #             imagebox = offsetbox.AnnotationBbox(
# #                 offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
# #                 X[i])
# #             ax.add_artist(imagebox)
# #     plt.xticks([]), plt.yticks([])
# #     if title is not None:
#         plt.title(title)


# def plot_with_labels(lowDWeights, labels):
def plot_with_labels(X, labels):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})



def run_model(domain_name, dataloaders, model, device, criterion):
    test_loss = 0.0
    test_acc = 0.0
    outputs = []
    outlab = []
    tlab = []

    for inputs, labels in dataloaders[domain_name]:

        if args.img_channel == 1:
            inputs = inputs.to(device)  # 对1通道图片的处理
        else:
            inputs = inputs.permute(0, 3, 1, 2).float().to(device)  # 对3通道图片的处理

        labels = labels.to(device)
        with torch.no_grad():
            logits = model(inputs)  # 对于inceptionV3网络时也采用同样的处理
            loss = criterion(logits, labels)
            outputs.append(logits.cpu().numpy())

            pred = logits.argmax(dim=1)
            outlab.append(pred.cpu().numpy())

            tlab.append(labels.cpu().numpy())  # 应该用真实的labels

            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)

            test_loss += loss_temp
            test_acc += correct

    test_loss = test_loss / len(dataloaders[domain_name].dataset)
    test_acc = test_acc / len(dataloaders[domain_name].dataset)

    logging.info('{} loss {}, {} acc {}'.format(domain_name, domain_name, test_loss, test_acc))

    print(len(dataloaders[domain_name].dataset))

    out = outputs[0]
    lab = outlab[0]
    tlb = tlab[0]
    for i in range(len(outputs) - 1):
        out = np.vstack((out, outputs[i + 1]))
        lab = np.hstack((lab, outlab[i + 1]))
        tlb = np.hstack((tlb, tlab[i + 1]))  # 应该用真实的labels

    print(out.shape)
    print(lab.shape)
    print(tlb.shape)

    tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=300)
    low_dim_embs = tsne.fit_transform(out)
    # plot_with_labels(low_dim_embs, lab)
    plot_with_labels(low_dim_embs, tlb)  # 应该用真实的labels





def t_sne(args, save_dir, pth_name):

    # device initial
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert args.batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))

    # dataset inital
    if args.data_name == 'CWRU':

        if args.img_channel == 1:
            from datasets import single_c_cwru_stft as datasets  # 注意这边加载的是专门针对1通道图像的
        elif args.img_channel == 3:
            # 注意这边加载的是专门针对3通道图像的____________________________________________________________
            from datasets import multi_c_cwru_stft as datasets
        else:
            raise Exception("image channel worry, not 1 or 3")

        Dataset = getattr(datasets, args.datasets_type)

    elif args.datasets == 'PU':
        pass
    else:
        raise Exception("processing type not implement")
    datasets = {}
    datasets['source_train'], datasets['source_val'], datasets['source_test'], datasets[
        'target_val'] \
        = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=False)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=(True if x.split('_')[1] == 'train' else False),
                                                  num_workers=args.num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['source_train', 'source_val', 'source_test', 'target_val']}

    # load model
    logging.info("___________using source best model______________________________")
    loadfilename = os.path.join(save_dir, '{}.pth'.format(pth_name))
    checkpoint = torch.load(loadfilename)
    model = getattr(models, args.model_name)(args.pretrained)
    model.load_state_dict(checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # run test
#     test_loss = 0.0
#     test_acc = 0.0
#     outputs = []
#     outlab = []
#     tlab = []
#
#     for inputs, labels in dataloaders['source_test']:
#
#         if args.img_channel == 1:
#             inputs = inputs.to(device)  # 对1通道图片的处理
#         else:
#             inputs = inputs.permute(0, 3, 1, 2).float().to(device)  # 对3通道图片的处理
#
#         labels = labels.to(device)
#         with torch.no_grad():
#             logits = model(inputs)  # 对于inceptionV3网络时也采用同样的处理
#             loss = criterion(logits, labels)
#             outputs.append(logits.cpu().numpy())
#
#             pred = logits.argmax(dim=1)
#             outlab.append(pred.cpu().numpy())
#
#             tlab.append(labels.cpu().numpy()) # 应该用真实的labels
#
#             correct = torch.eq(pred, labels).float().sum().item()
#             loss_temp = loss.item() * inputs.size(0)
#
#             test_loss += loss_temp
#             test_acc += correct
#
#     test_loss = test_loss / len(dataloaders['source_test'].dataset)
#     test_acc = test_acc / len(dataloaders['source_test'].dataset)
#
#     logging.info('test loss {}, test acc {}'.format(test_loss, test_acc))
#
#     print(len(dataloaders['source_test'].dataset))
#
#     out = outputs[0]
#     lab = outlab[0]
#     tlb = tlab[0]
#     for i in range(len(outputs)-1):
#         out = np.vstack((out, outputs[i+1]))
#         lab = np.hstack((lab, outlab[i + 1]))
#         tlb = np.hstack((tlb, tlab[i+1])) # 应该用真实的labels
#
#     print(out.shape)
#     print(lab.shape)
#     print(tlb.shape)
#
#     tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=300)
#     low_dim_embs = tsne.fit_transform(out)
#     # plot_with_labels(low_dim_embs, lab)
#     plot_with_labels(low_dim_embs, tlb)  # 应该用真实的labels
#
#
# #________________迁移后，在目标数据集上的t-sne——————————————————————————————
#     ttest_loss = 0.0
#     ttest_acc = 0.0
#     toutputs = []
#     toutlab = []
#     ttlab = []
#
#     for inputs, labels in dataloaders['target_val']:
#
#         if args.img_channel == 1:
#             inputs = inputs.to(device)  # 对1通道图片的处理
#         else:
#             inputs = inputs.permute(0, 3, 1, 2).float().to(device)  # 对3通道图片的处理
#
#         labels = labels.to(device)
#         with torch.no_grad():
#             logits = model(inputs)
#             loss = criterion(logits, labels)
#
#             toutputs.append(logits.cpu().numpy())
#
#             pred = logits.argmax(dim=1)
#
#             toutlab.append(pred.cpu().numpy())
#
#             ttlab.append(labels.cpu().numpy())  # 应该用真实的labels
#
#             correct = torch.eq(pred, labels).float().sum().item()
#             loss_temp = loss.item() * inputs.size(0)
#
#             ttest_loss += loss_temp
#             ttest_acc += correct
#
#     ttest_loss = ttest_loss / len(dataloaders['target_val'].dataset)
#     ttest_acc = ttest_acc / len(dataloaders['target_val'].dataset)
#
#     logging.info('target_val loss {}, target_val acc {}'.format(ttest_loss, ttest_acc))
#
#     print(len(dataloaders['target_val'].dataset))
#
#     tout = toutputs[0]
#     tlab = toutlab[0]
#     ttlb = ttlab[0]
#     for i in range(len(toutputs) - 1):
#         tout = np.vstack((tout, toutputs[i + 1]))
#         tlab = np.hstack((tlab, toutlab[i + 1]))
#         ttlb = np.hstack((ttlb, ttlab[i + 1]))  # 应该用真实的labels
#
#     print(tout.shape)
#     print(tlab.shape)
#     print(ttlb.shape)
#
#     tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=300)
#     tlow_dim_embs = tsne.fit_transform(tout)
#     plot_with_labels(tlow_dim_embs, ttlb)  # 应该用真实的labels

    run_model('source_test', dataloaders, model, device, criterion)










#___________________________________________________________test code__________________________________
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

    parser.add_argument('--img_channel', type=int, default=3, help='input images channel, 3 or 1')
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')

    parser.add_argument('--data_dir', type=str, default="E:\PycharmProjects\data\CWRU",
                        help='the directory of the data')
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
    parser.add_argument('--checkpoint_dir', type=str, default='E:/PycharmProjects/paper1_changechanneldone/checkpoint',
                        help='the directory to save the model')
    # parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    # parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
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
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    return args

#____________test______
args = parse_args()
# sub_dir = args.model_name + '_' + args.datasets_type + '_' + '0924-110927'
# sub_dir = args.model_name + '_' + args.datasets_type + '_' + '0924-115530'
# sub_dir = args.model_name + '_' + args.datasets_type + '_' + '3c2ep'
sub_dir = args.model_name + '_' + args.datasets_type + '_' + '3c2epf'


save_dir = os.path.join(args.checkpoint_dir, sub_dir)


# pth_name = '9-0.9856-best_model'
# pth_name = '0-0.3636-best_model'
# pth_name = '1-0.6268-best_model'
pth_name = '0-1.0000-best_model'
# pth_name = 'target fine-tune 0.9739-last_model'


setlogger(os.path.join(save_dir, 'test.log'))
t_sne(args, save_dir, pth_name)



plt.show()





