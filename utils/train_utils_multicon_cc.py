import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import models


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        self.best_acc = 0.0
        self.best_epoch = 0
        # self.last_acc = 0.0




    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))



        # 我的修改
        if args.data_name == 'CWRU':

            if args.img_channel == 1:
                from datasets import single_c_cwru_stft as datasets   #注意这边加载的是专门针对1通道图像的
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


        self.datasets = {}



        if isinstance(args.transfer_task[0], str):
           #print( args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))


        if args.finetune == True:
            self.datasets['source_train'], self.datasets['source_val'], self.datasets['source_test'], \
            self.datasets['target_train'], self.datasets['target_val'] = \
                Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                               shuffle=(True if x.split('_')[1] == 'train' else False),
                                                               num_workers=args.num_workers,
                                                               pin_memory=(True if self.device == 'cuda' else False))
                                for x in ['source_train', 'source_val', 'source_test', 'target_train', 'target_val']}

            # self.model_eval = getattr(models, args.model_name)(args.pretrained)
            # self.model_eval.fc = torch.nn.Linear(self.model_eval.fc.in_features, Dataset.num_classes)

        else:
            self.datasets['source_train'], self.datasets['source_val'], self.datasets['source_test'], self.datasets['target_val'] \
            = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=False)

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'source_val', 'source_test', 'target_val']}

        # Define the model
        # 实例化自编的models类来生成模型
        self.model = getattr(models, args.model_name)(args.pretrained)





        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.adabn:
                self.model_eval = torch.nn.DataParallel(self.model_eval)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    # print('\t', name)
                    logging.info('param.requires_grad == True  {}'.format(name))

            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay) # fine-turning 会对所有的参数进行微调
        else:
            raise Exception("optimizer not implement")



        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")



        # Load the checkpoint
        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        #
        # if args.adabn:
        #     self.model_eval.to(self.device)


        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        total_train_times = 0
        begin_time = time.time()


        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))



            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'source_test']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    #通道合并从这里下手
                    if args.img_channel == 1:
                        inputs = inputs.to(self.device)  #对1通道图片的处理
                    else:
                        inputs = inputs.permute(0, 3, 1, 2).float().to(self.device)  #对3通道图片的处理



                    labels = labels.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'source_train'):

                        # forward
                        if args.model_name == "inception":
                            if phase == 'source_train':
                                # logits, aux_logits = self.model(inputs)
                                logits = self.model(inputs)
                            else:
                                logits = self.model(inputs)
                        else:
                            logits = self.model(inputs)

                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time # 传入的参数args.print_step
                                sample_per_sec = 1.0*batch_count/train_time

                                total_train_times += train_time

                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
                ))

                # save the model
                if phase == 'source_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                    # save the best model according to the val accuracy
                    # if epoch_acc > best_acc or epoch > args.max_epoch-2:
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}. ******************************".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                        self.best_acc = best_acc
                        self.best_epoch = epoch

                    # save the last model
                    if epoch == args.max_epoch-1:
                        logging.info("save last model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-last_model.pth'.format(epoch, epoch_acc)))



            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info("total times {:.4f} sec, total train times {:.4f} sec".format(time.time()-begin_time, total_train_times))




    def transfer_condition(self, savemodelpath=None, testmodel='best'): # savemodelpath应该有两个分别是last，和best
        args = self.args
        testmodel = args.testmodel

        if savemodelpath is not None:
            loadfilename = os.path.join(self.save_dir, savemodelpath)
            checkpoint = torch.load(loadfilename)
            self.model.load_state_dict(checkpoint)

        elif testmodel == 'best': # testmodel也有两个分别是best和last，默认为last，因为不需要再读取模型了
            logging.info("___________using source best model______________________________")
            loadfilename = os.path.join(self.save_dir,
                                                '{}-{:.4f}-best_model.pth'.format(self.best_epoch, self.best_acc))
            checkpoint = torch.load(loadfilename)

            if args.finetune == True:
                self.model.load_state_dict(checkpoint)
                for param in self.model.parameters():
                    param.requires_grad = False

                for param in self.model.fc.parameters():
                    param.requires_grad = True

                # self.model.fc = nn.Linear(self.model.fc.in_features, self.model.fc.out_features).to(self.device)

                for name, param in self.model.named_parameters():
                    if param.requires_grad == True:
                        # print('\t', name)
                        logging.info('param.requires_grad == True  {}'.format(name))

            else:
                self.model.load_state_dict(checkpoint)
        else:
            pass

        target_val_acc = 0.0
        target_val_loss = 0.0
        target_train_acc = 0.0
        target_train_loss = 0.0

        logging.info("___________target domain______________________________")

        if args.finetune == True:
            logging.info("___________fine-tune______________________________")

            for epoch in range(args.target_train_epoch):

                for inputs, labels in self.dataloaders['target_train']:

                    if args.img_channel == 1:
                        inputs = inputs.to(self.device)  # 对1通道图片的处理
                    else:
                        inputs = inputs.permute(0, 3, 1, 2).float().to(self.device)  # 对3通道图片的处理


                    labels = labels.to(self.device)
                    with torch.set_grad_enabled(True):

                        if args.model_name == "inception":
                            # logits, aux_logits = self.model(inputs)
                            logits = self.model(inputs)
                        else:
                            logits = self.model(inputs)

                        # logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        target_train_loss += loss_temp
                        target_train_acc += correct

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()


                target_train_loss = target_train_loss / len(self.dataloaders['target_train'].dataset)
                target_train_acc = target_train_acc / len(self.dataloaders['target_train'].dataset)

                model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                torch.save(model_state_dic,
                           os.path.join(self.save_dir, 'target fine-tune {:.4f}-last_model.pth'.format(target_train_acc)))

                logging.info(
                        "last target domain fine tune model, epoch {} target train loss {} and target train acc {:.4f}".format(epoch, target_train_loss, target_train_acc))
        else:
            pass


        for inputs, labels in self.dataloaders['target_val']:

            if args.img_channel == 1:
                inputs = inputs.to(self.device)  # 对1通道图片的处理
            else:
                inputs = inputs.permute(0, 3, 1, 2).float().to(self.device)  # 对3通道图片的处理

            # inputs = inputs.to(self.device)  # 对1通道图片的处理
            # inputs = inputs.permute(0, 3, 1, 2).float().to(self.device)  # 对3通道图片的处理

            labels = labels.to(self.device)
            with torch.no_grad():

                logits = self.model(inputs) # 对于inceptionV3网络时也采用同样的处理

                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()
                loss_temp = loss.item() * inputs.size(0)

                target_val_loss += loss_temp
                target_val_acc += correct

        target_val_loss = target_val_loss / len(self.dataloaders['target_val'].dataset)
        target_val_acc = target_val_acc / len(self.dataloaders['target_val'].dataset)


        if args.finetune:
            logging.info("last model target test loss {} and target test acc {:.4f}".format(target_val_loss, target_val_acc))
        elif testmodel == 'best':
            logging.info("best source model target val loss {} and target val acc {:.4f}".format(target_val_loss,  target_val_acc))
        else:
            logging.info("last source model target val loss {} and target val acc {:.4f}".format(target_val_loss, target_val_acc))





