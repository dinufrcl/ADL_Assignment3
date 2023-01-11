import torchvision.models as models
import torch
from torchsummary import summary
import timm

class UpsamplingBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, s_channels):
        super(UpsamplingBlock, self).__init__()

        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel+s_channels, input_channel+s_channels, 3, padding=1, bias=False), # channels?
            torch.nn.BatchNorm2d(input_channel+s_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(input_channel+s_channels, output_channel, 1)
        )


    def forward(self,x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)

        return x

# class HRNetw32Backbone(torch.nn.Module):
#     def __init__(self):
#         super(HRNetw32Backbone, self).__init__()
#         self.backbone = timm.create_model('hrnet_w32', pretrained=True)
#
#     def forward(self,x):
#         # Stem
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.act1(x)
#         x = self.backbone.conv2(x)
#         x = self.backbone.bn2(x)
#         x = self.backbone.act2(x)
#
#         # Stages
#         yl = self.backbone.stages(x)
#         return yl[0]


class HRNetw32Model(torch.nn.Module):
    def __init__(self):
        super(HRNetw32Model, self).__init__()
        self.backbone = timm.create_model('hrnet_w32', pretrained=True)

        # todo: PCK schwankt sehr stark --> liegt am Modell oder Umrechnung der Koordinaten?

        # self.basic_block = torch.nn.Sequential(
        #     torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.ReLU()
        # )

        self.basic_block = models.resnet.BasicBlock(32, 32)

        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.out = torch.nn.Conv2d(32, 17, 1)

    def backbone_func(self, x):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        # Stages
        yl = self.backbone.stages(x)
        return yl[0]

    def head(self, x):
        x = self.basic_block(x)
        x = self.up(x)
        x = self.basic_block(x)
        x = self.out(x)

        return x

    def forward(self, x):
        x = self.backbone_func(x)
        x = self.head(x)
        return x


class HRNetw32ASPPModel(torch.nn.Module):
    def __init__(self):
        super(HRNetw32ASPPModel, self).__init__()
        self.backbone = timm.create_model('hrnet_w32', pretrained=True)

        self.aspp = ASPP(32, 32)

        # self.basic_block = torch.nn.Sequential(
        #     torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.ReLU()
        # )

        self.basic_block = models.resnet.BasicBlock(32, 32)

        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.out = torch.nn.Conv2d(32, 17, 1)

    def backbone_func(self, x):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        # Stages
        yl = self.backbone.stages(x)
        return yl[0]

    def head(self, x):
        x = self.aspp.forward(x)
        x = self.basic_block(x)
        x = self.up(x)
        x = self.basic_block(x)
        x = self.out(x)

        return x

    def forward(self, x):
        x = self.backbone_func(x)
        x = self.head(x)
        return x


#
# class ResNet18Backbone(torch.nn.Module):
#     def __init__(self):
#         super(ResNet18Backbone, self).__init__()
#         self.backbone = models.resnet18(pretrained=True)
#
#     def forward(self, x):
#         skip_connections = []
#         x = self.backbone.conv1(x)
#         # 128
#         x = self.backbone.relu(self.backbone.bn1(x))
#         skip_connections.append(x)
#         x = self.backbone.maxpool(x)
#         # 64
#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         # 32
#         x = self.backbone.layer3(x)
#         # 16
#         x = self.backbone.layer4(x)
#         # 8
#
#         return skip_connections, x
#
#

class ResNet18Model(torch.nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)

        # self.aspp = ASPP(512, 512)

        self.up_block1 = UpsamplingBlock(input_channel=512, output_channel=256, s_channels=256)
        self.up_block2 = UpsamplingBlock(input_channel=256, output_channel=128, s_channels=128)
        self.up_block3 = UpsamplingBlock(input_channel=128, output_channel=64, s_channels=64)
        self.up_block4 = UpsamplingBlock(input_channel=64, output_channel=64, s_channels=64)

        self.output_layer = torch.nn.Sequential(
                                                torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                                torch.nn.BatchNorm2d(64),
                                                torch.nn.ReLU(),
                                                torch.nn.Conv2d(64, 17, 1)  # map channel dims to correct amount of classes
                                                )

    def backbone_func(self, x):
        skip_connections = []
        x = self.backbone.conv1(x)
        x = self.backbone.relu(self.backbone.bn1(x))
        skip_connections.append(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        skip_connections.append(x)
        x = self.backbone.layer2(x)
        skip_connections.append(x)
        x = self.backbone.layer3(x)
        skip_connections.append(x)
        x = self.backbone.layer4(x)

        return skip_connections, x

    def head(self, x, s):
        # x = self.aspp.forward(x)
        x = self.up_block1.forward(x, s[3])
        x = self.up_block2.forward(x, s[2])
        x = self.up_block3.forward(x, s[1])
        x = self.up_block4.forward(x, s[0])
        x = self.output_layer(x)

        return x

    def forward(self, x):
        s, x = self.backbone_func(x)
        x = self.head(x, s)

        return x


class ResNet18ASPPModel(torch.nn.Module):
    def __init__(self):
        super(ResNet18ASPPModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)

        self.aspp = ASPP(512, 512)

        self.up_block1 = UpsamplingBlock(input_channel=512, output_channel=256, s_channels=256)
        self.up_block2 = UpsamplingBlock(input_channel=256, output_channel=128, s_channels=128)
        self.up_block3 = UpsamplingBlock(input_channel=128, output_channel=64, s_channels=64)
        self.up_block4 = UpsamplingBlock(input_channel=64, output_channel=64, s_channels=64)

        self.output_layer = torch.nn.Sequential(
                                                torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                                torch.nn.BatchNorm2d(64),
                                                torch.nn.ReLU(),
                                                torch.nn.Conv2d(64, 17, 1)  # map channel dims to correct amount of classes
                                                )

    def backbone_func(self, x):
        skip_connections = []
        x = self.backbone.conv1(x)
        x = self.backbone.relu(self.backbone.bn1(x))
        skip_connections.append(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        skip_connections.append(x)
        x = self.backbone.layer2(x)
        skip_connections.append(x)
        x = self.backbone.layer3(x)
        skip_connections.append(x)
        x = self.backbone.layer4(x)

        return skip_connections, x

    def head(self, x, s):
        x = self.aspp.forward(x)
        x = self.up_block1.forward(x, s[3])
        x = self.up_block2.forward(x, s[2])
        x = self.up_block3.forward(x, s[1])
        x = self.up_block4.forward(x, s[0])
        x = self.output_layer(x)

        return x

    def forward(self, x):
        s, x = self.backbone_func(x)
        x = self.head(x, s)

        return x

class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        atrous_rates = [4, 8, 12, 16]

        modules = []
        modules.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU())
        )

        for rate in atrous_rates:
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU()))

        self.aspp_convs = torch.nn.ModuleList(modules)

        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(len(self.aspp_convs) * out_channels, out_channels, 1, bias=False),  # concat + 1x1
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )

    def forward(self, x):
        _res = []
        for conv in self.aspp_convs:
            _res.append(conv(x))
        x = torch.cat(_res, dim=1)
        x = self.project(x)

        return x


if __name__ == "__main__":
    # model = ResNet18Model()
    model = ResNet18Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    summary(model, (3, 128, 128))


