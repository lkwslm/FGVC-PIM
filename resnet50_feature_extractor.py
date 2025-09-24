"""
ResNet50分类模型四个阶段特征提取器
提取ResNet50的conv2_x, conv3_x, conv4_x, conv5_x四个阶段的特征
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from transformers import ResNetModel, AutoImageProcessor


class ResNet50FeatureExtractor:
    """ResNet50四个阶段特征提取器"""
    
    def __init__(self, pretrained: bool = True, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.features = {}
        self.model = None
        self.processor = None
        self.load_model(pretrained)
        self.register_hooks()
    
    def load_model(self, pretrained: bool = True):
        """加载ResNet50模型"""
        print("正在加载ResNet50模型...")
        
        # 方法1: 使用torchvision的ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # 定义预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ ResNet50模型加载完成")
    
    def load_huggingface_resnet(self, model_name: str = "microsoft/resnet-50"):
        """使用HuggingFace的ResNet50模型"""
        print(f"正在加载HuggingFace ResNet50: {model_name}")
        
        self.model = ResNetModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ HuggingFace ResNet50模型加载完成")
    
    def register_hooks(self):
        """注册钩子函数以提取四个阶段的特征"""
        self.features = {}
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach().cpu()
            return hook
        
        # ResNet50的四个主要阶段
        # Stage 1: conv2_x (layer1) - 输出尺寸: 56x56, 通道数: 256
        # Stage 2: conv3_x (layer2) - 输出尺寸: 28x28, 通道数: 512  
        # Stage 3: conv4_x (layer3) - 输出尺寸: 14x14, 通道数: 1024
        # Stage 4: conv5_x (layer4) - 输出尺寸: 7x7, 通道数: 2048
        
        if hasattr(self.model, 'layer1'):  # torchvision ResNet
            self.model.layer1.register_forward_hook(get_hook('stage1_conv2x'))
            self.model.layer2.register_forward_hook(get_hook('stage2_conv3x'))
            self.model.layer3.register_forward_hook(get_hook('stage3_conv4x'))
            self.model.layer4.register_forward_hook(get_hook('stage4_conv5x'))
            
            # 也可以提取更细粒度的特征
            self.model.conv1.register_forward_hook(get_hook('conv1'))
            self.model.bn1.register_forward_hook(get_hook('bn1'))
            self.model.avgpool.register_forward_hook(get_hook('avgpool'))
            
        elif hasattr(self.model, 'encoder'):  # HuggingFace ResNet
            # HuggingFace ResNet的层结构稍有不同
            if hasattr(self.model.encoder, 'stages'):
                for i, stage in enumerate(self.model.encoder.stages):
                    stage.register_forward_hook(get_hook(f'stage{i+1}'))
        
        print("✓ 已注册四个阶段的特征提取钩子")
    
    def extract_features(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        提取图像的四个阶段特征
        
        Args:
            image: PIL图像
            
        Returns:
            包含四个阶段特征的字典
        """
        self.features = {}
        
        if self.processor is not None:  # HuggingFace模型
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:  # torchvision模型
            # 预处理图像
            if isinstance(image, Image.Image):
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            else:
                input_tensor = image.to(self.device)
            
            # 前向传播
            with torch.no_grad():
                output = self.model(input_tensor)
                self.features['final_output'] = output.cpu()
        
        return self.features
    
    def extract_features_from_path(self, image_path: str) -> Dict[str, torch.Tensor]:
        """从图像路径提取特征"""
        image = Image.open(image_path).convert('RGB')
        return self.extract_features(image)
    
    def get_feature_info(self) -> Dict[str, Dict]:
        """获取各阶段特征的详细信息"""
        if not self.features:
            return {"error": "请先提取特征"}
        
        info = {}
        for name, feature in self.features.items():
            if isinstance(feature, torch.Tensor):
                info[name] = {
                    "shape": list(feature.shape),
                    "dtype": str(feature.dtype),
                    "min_value": float(feature.min()),
                    "max_value": float(feature.max()),
                    "mean_value": float(feature.mean()),
                    "std_value": float(feature.std())
                }
        
        return info
    
    def visualize_feature_maps(self, stage_name: str, num_channels: int = 16, 
                              save_path: Optional[str] = None):
        """
        可视化指定阶段的特征图
        
        Args:
            stage_name: 阶段名称
            num_channels: 要显示的通道数
            save_path: 保存路径
        """
        if stage_name not in self.features:
            print(f"错误: 未找到阶段 {stage_name}")
            return
        
        feature = self.features[stage_name]
        if len(feature.shape) != 4:  # [batch, channel, height, width]
            print(f"错误: 特征形状不正确 {feature.shape}")
            return
        
        # 取第一个batch的特征
        feature = feature[0]  # [channel, height, width]
        num_channels = min(num_channels, feature.shape[0])
        
        # 创建子图
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'{stage_name} Feature Maps (Shape: {feature.shape})')
        
        for i in range(16):
            row, col = i // 4, i % 4
            if i < num_channels:
                # 显示特征图
                feature_map = feature[i].numpy()
                axes[row, col].imshow(feature_map, cmap='viridis')
                axes[row, col].set_title(f'Channel {i}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 特征图已保存到: {save_path}")
        
        plt.show()
    
    def save_features(self, save_path: str):
        """保存提取的特征"""
        if not self.features:
            print("错误: 没有可保存的特征")
            return
        
        torch.save(self.features, save_path)
        print(f"✓ 特征已保存到: {save_path}")
    
    def compare_stages(self):
        """比较四个阶段的特征统计信息"""
        if not self.features:
            print("错误: 请先提取特征")
            return
        
        print("\n=== ResNet50四个阶段特征对比 ===")
        print(f"{'阶段':<15} {'形状':<20} {'通道数':<8} {'空间尺寸':<12} {'参数量':<12}")
        print("-" * 70)
        
        stage_info = {
            'stage1_conv2x': 'Stage1(conv2_x)',
            'stage2_conv3x': 'Stage2(conv3_x)', 
            'stage3_conv4x': 'Stage3(conv4_x)',
            'stage4_conv5x': 'Stage4(conv5_x)'
        }
        
        for key, name in stage_info.items():
            if key in self.features:
                feature = self.features[key]
                shape = feature.shape
                channels = shape[1] if len(shape) > 1 else 0
                spatial = f"{shape[2]}x{shape[3]}" if len(shape) > 3 else "N/A"
                params = channels * (shape[2] * shape[3] if len(shape) > 3 else 1)
                
                print(f"{name:<15} {str(shape):<20} {channels:<8} {spatial:<12} {params:<12}")


def demo_resnet50_feature_extraction():
    """ResNet50特征提取演示"""
    
    print("=== ResNet50四个阶段特征提取演示 ===\n")
    
    # 创建特征提取器
    extractor = ResNet50FeatureExtractor(pretrained=True)
    
    # 创建示例图像或使用现有图像
    try:
        # 尝试使用项目中的图像
        image_path = "imgs/0001.png"
        image = Image.open(image_path).convert('RGB')
        print(f"使用图像: {image_path}")
    except:
        # 创建随机图像作为示例
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("使用随机生成的示例图像")
    
    # 提取特征
    print("\n正在提取四个阶段特征...")
    features = extractor.extract_features(image)
    
    # 显示特征信息
    print(f"\n✓ 成功提取 {len(features)} 个特征层:")
    for name, feature in features.items():
        if isinstance(feature, torch.Tensor):
            print(f"  {name}: {feature.shape}")
    
    # 获取详细信息
    print("\n=== 特征详细信息 ===")
    info = extractor.get_feature_info()
    for name, details in info.items():
        if isinstance(details, dict):
            print(f"\n{name}:")
            print(f"  形状: {details['shape']}")
            print(f"  数值范围: [{details['min_value']:.4f}, {details['max_value']:.4f}]")
            print(f"  均值: {details['mean_value']:.4f}")
            print(f"  标准差: {details['std_value']:.4f}")
    
    # 比较各阶段
    extractor.compare_stages()
    
    # 可视化第一个阶段的特征图（如果有matplotlib）
    try:
        if 'stage1_conv2x' in features:
            print("\n正在生成Stage1特征图可视化...")
            extractor.visualize_feature_maps('stage1_conv2x', 
                                           save_path='vis/resnet50_stage1_features.png')
    except Exception as e:
        print(f"特征图可视化失败: {e}")
    
    # 保存特征
    try:
        extractor.save_features('resnet50_features.pth')
    except Exception as e:
        print(f"保存特征失败: {e}")
    
    return extractor, features


def load_and_extract_custom_resnet():
    """加载自定义ResNet模型并提取特征的示例"""
    
    print("\n=== 自定义ResNet模型特征提取 ===")
    
    # 如果你有自定义的ResNet模型文件
    """
    # 示例：加载自定义模型
    extractor = ResNet50FeatureExtractor(pretrained=False)
    
    # 加载自定义权重
    checkpoint = torch.load('path/to/your/resnet50_checkpoint.pth')
    extractor.model.load_state_dict(checkpoint['model_state_dict'])
    
    # 重新注册钩子（因为模型结构可能有变化）
    extractor.register_hooks()
    
    # 提取特征
    image = Image.open('your_image.jpg')
    features = extractor.extract_features(image)
    """
    
    print("请参考注释中的代码来加载自定义ResNet模型")


if __name__ == "__main__":
    # 运行演示
    extractor, features = demo_resnet50_feature_extraction()
    
    # 如果需要加载自定义模型
    # load_and_extract_custom_resnet()
    
    print("\n=== 使用建议 ===")
    print("1. Stage1(conv2_x): 低级特征，边缘、纹理等")
    print("2. Stage2(conv3_x): 中级特征，形状、模式等") 
    print("3. Stage3(conv4_x): 高级特征，对象部件等")
    print("4. Stage4(conv5_x): 最高级特征，语义信息等")
    print("\n根据你的任务需求选择合适的阶段特征进行使用。")