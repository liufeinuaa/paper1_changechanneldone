import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# -----------------------input size>=111*111---------------------------------
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, in_channel=3, out_channel=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.my_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, out_channel),
        )
        self.fc = nn.Linear(4096, out_channel)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = x.reshape(x.size(0), 256 * 6 * 6)
        x = self.my_classifier(x)
        x = self.fc(x)
        return x

def alexnet(pretrained, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = AlexNet(**kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    # return model

    # model = models.AlexNet(pretrained, progress=True)
    # if pretrained == True:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    #
    # num_fc_inputs = model.classifier.6.in_features
    # model.fc = nn.Linear(num_fc_inputs, 10)
    #
    # return model

    model = AlexNet(**kwargs)

    if pretrained == True:
        pre_model = models.alexnet(pretrained, progress=True)
        # pre_model.load_state_dict(torch.load('./models/resnet18-5c106cde.pth'))  # 指定路径加载
        pre_dict = pre_model.state_dict()

        for params in pre_dict.keys():
            if params in model.state_dict():
                model.state_dict()[params].copy_(pre_dict[params])

        for param in model.parameters():
            param.requires_grad = False

        for param in model.my_classifier.parameters():
            param.requires_grad = True

    num_fc_inputs = model.fc.in_features
    model.fc = nn.Linear(num_fc_inputs, 10)

    return model





