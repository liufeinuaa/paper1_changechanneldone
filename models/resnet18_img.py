import torch
import torchvision.models as models
import torch.nn as nn



def resnet18(pretrained, **kwargs):

    # if pretrained == False:
    #     model = models.resnet18(pretrained=)

    model = models.resnet18(pretrained, progress=True)
    # num_fc_inputs = model.fc.in_features
    # model.fc = nn.Linear(num_fc_inputs, 10)

    if pretrained == True:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        # for param in model.layer4[1].parameters(): #修改更少的参数
        #     param.requires_grad = True


    num_fc_inputs = model.fc.in_features
    model.fc = nn.Linear(num_fc_inputs, 10)

    return model









