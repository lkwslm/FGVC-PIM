import torch
import torch.nn as nn
from transformers import ResNetForImageClassification
from .pim_module.pim_module import PluginMoodel

class ResNetBackbone(nn.Module):
    """
    自定义 ResNet backbone，用于返回中间层特征
    """
    def __init__(self, pretrained_path=None):
        super().__init__()
        # 加载 Hugging Face ResNet 模型
        if pretrained_path:
            self.resnet_model = ResNetForImageClassification.from_pretrained(pretrained_path)
        else:
            self.resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        
        # 获取 ResNet backbone (不包含分类头)
        self.backbone = self.resnet_model.resnet
        
    def forward(self, x):
        # 通过 embedder (conv + pooling)
        x = self.backbone.embedder(x)
        
        # 通过各个 stage，收集所有中间特征
        features = {}
        for i, stage in enumerate(self.backbone.encoder.stages):
            x = stage(x)
            # 收集所有 stage 的特征
            if i == 0:  # stage 0 输出: [B, 256, 56, 56]
                features['layer0'] = x
            elif i == 1:  # stage 1 输出: [B, 512, 28, 28]
                features['layer1'] = x
            elif i == 2:  # stage 2 输出: [B, 1024, 14, 14]
                features['layer2'] = x
            elif i == 3:  # stage 3 输出: [B, 2048, 7, 7]
                features['layer3'] = x
        
        return features

class ResNetPIM(nn.Module):
    def __init__(self, num_classes=2, pretrain = True, pretrained_path=None):
        super().__init__()
        
        # 如果没有指定预训练路径，使用本地预训练模型
        if pretrain:
            pretrained_path = "E:\\Code\\FGVC-PIM\\pretrained_models\\resnet50"
        
        # 创建自定义 backbone
        self.backbone = ResNetBackbone(pretrained_path)
        
        # 由于我们使用自定义 backbone 返回字典，设置 return_nodes 为 None
        return_nodes = None

        # 定义 PIM 模型的参数
        # 根据实际的特征维度调整 num_selects，支持所有 4 个 stage
        num_selects = {
            "layer0": 256,  # 从 256 通道中选择 256 个
            "layer1": 128,  # 从 512 通道中选择 128 个
            "layer2": 64,   # 从 1024 通道中选择 64 个
            "layer3": 32    # 从 2048 通道中选择 32 个
        }
        
        IMG_SIZE = 480
        USE_FPN = True
        FPN_SIZE = 256
        PROJ_TYPE = "Conv"
        UPSAMPLE_TYPE = "Bilinear"

        # 创建 PIM 模型
        self.pim_model = PluginMoodel(
            backbone=self.backbone,
            return_nodes=return_nodes,
            img_size=IMG_SIZE,
            use_fpn=USE_FPN,
            fpn_size=FPN_SIZE,
            proj_type=PROJ_TYPE,
            upsample_type=UPSAMPLE_TYPE,
            use_selection=True,
            num_classes=num_classes,
            num_selects=num_selects,
            use_combiner=True,
            comb_proj_size=None
        )

    def forward(self, x):
        return self.pim_model(x)


if __name__ == "__main__":
    # 使用默认的本地预训练模型路径
    model = ResNetPIM(num_classes=2)
    # print(model)