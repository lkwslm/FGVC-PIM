"""
HuggingFace预训练模型加载和中间特征提取示例
支持.bin和.safetensor格式的模型文件
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    ViTModel, ViTImageProcessor,
    ResNetModel, AutoImageProcessor,
    BertModel, BertTokenizer
)
from safetensors.torch import load_file
import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image


class HuggingFaceModelLoader:
    """HuggingFace模型加载器，支持多种格式和特征提取"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.intermediate_features = {}
        
    def load_model_from_bin(self, model_path: str, config_path: Optional[str] = None) -> None:
        """
        从.bin文件加载预训练模型
        
        Args:
            model_path: .bin模型文件路径
            config_path: 配置文件路径（可选）
        """
        print(f"正在从.bin文件加载模型: {model_path}")
        
        # 加载配置
        if config_path:
            self.config = AutoConfig.from_pretrained(config_path)
        else:
            # 尝试从模型目录加载配置
            model_dir = os.path.dirname(model_path)
            self.config = AutoConfig.from_pretrained(model_dir)
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 创建模型实例
        self.model = AutoModel.from_config(self.config)
        
        # 加载权重
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载对应的tokenizer/processor
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))
        except:
            try:
                self.processor = AutoImageProcessor.from_pretrained(os.path.dirname(model_path))
            except:
                print("警告: 无法加载tokenizer或processor")
        
        print("✓ .bin模型加载完成")
    
    def load_model_from_safetensor(self, model_path: str, config_path: Optional[str] = None) -> None:
        """
        从.safetensor文件加载预训练模型
        
        Args:
            model_path: .safetensor模型文件路径
            config_path: 配置文件路径（可选）
        """
        print(f"正在从.safetensor文件加载模型: {model_path}")
        
        # 加载配置
        if config_path:
            self.config = AutoConfig.from_pretrained(config_path)
        else:
            model_dir = os.path.dirname(model_path)
            self.config = AutoConfig.from_pretrained(model_dir)
        
        # 使用safetensors加载权重
        state_dict = load_file(model_path)
        
        # 创建模型实例
        self.model = AutoModel.from_config(self.config)
        
        # 加载权重
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载对应的tokenizer/processor
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))
        except:
            try:
                self.processor = AutoImageProcessor.from_pretrained(os.path.dirname(model_path))
            except:
                print("警告: 无法加载tokenizer或processor")
        
        print("✓ .safetensor模型加载完成")
    
    def load_model_from_hub(self, model_name: str, use_safetensors: bool = True) -> None:
        """
        从HuggingFace Hub加载模型
        
        Args:
            model_name: 模型名称
            use_safetensors: 是否优先使用safetensors格式
        """
        print(f"正在从HuggingFace Hub加载模型: {model_name}")
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=use_safetensors,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # 加载配置
        self.config = self.model.config
        
        # 加载tokenizer或processor
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
            except:
                print("警告: 无法加载tokenizer或processor")
        
        self.model.eval()
        print("✓ Hub模型加载完成")
    
    def register_hooks_for_feature_extraction(self, layer_names: List[str] = None) -> None:
        """
        注册钩子函数以提取中间特征
        
        Args:
            layer_names: 要提取特征的层名称列表，如果为None则提取所有层
        """
        self.intermediate_features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # 如果输出是元组，通常第一个元素是主要输出
                    self.intermediate_features[name] = output[0].detach().cpu()
                else:
                    self.intermediate_features[name] = output.detach().cpu()
            return hook
        
        # 如果没有指定层名称，则为所有命名模块注册钩子
        if layer_names is None:
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # 只为叶子模块注册钩子
                    module.register_forward_hook(hook_fn(name))
        else:
            # 为指定的层注册钩子
            for name, module in self.model.named_modules():
                if name in layer_names:
                    module.register_forward_hook(hook_fn(name))
        
        print(f"✓ 已注册特征提取钩子")
    
    def extract_features_text(self, text: Union[str, List[str]], 
                            max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        提取文本的中间特征
        
        Args:
            text: 输入文本或文本列表
            max_length: 最大序列长度
            
        Returns:
            包含中间特征的字典
        """
        if self.tokenizer is None:
            raise ValueError("未加载tokenizer，无法处理文本输入")
        
        # 清空之前的特征
        self.intermediate_features = {}
        
        # 编码文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 添加最终输出
        if hasattr(outputs, 'last_hidden_state'):
            self.intermediate_features['final_output'] = outputs.last_hidden_state.cpu()
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            self.intermediate_features['pooled_output'] = outputs.pooler_output.cpu()
        
        return self.intermediate_features
    
    def extract_features_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor, List]) -> Dict[str, torch.Tensor]:
        """
        提取图像的中间特征
        
        Args:
            image: 输入图像
            
        Returns:
            包含中间特征的字典
        """
        if self.processor is None:
            raise ValueError("未加载image processor，无法处理图像输入")
        
        # 清空之前的特征
        self.intermediate_features = {}
        
        # 处理图像
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 添加最终输出
        if hasattr(outputs, 'last_hidden_state'):
            self.intermediate_features['final_output'] = outputs.last_hidden_state.cpu()
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            self.intermediate_features['pooled_output'] = outputs.pooler_output.cpu()
        
        return self.intermediate_features
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型未加载"}
        
        info = {
            "model_type": self.config.model_type if self.config else "unknown",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": next(self.model.parameters()).device,
            "dtype": next(self.model.parameters()).dtype,
        }
        
        if self.config:
            info.update({
                "hidden_size": getattr(self.config, 'hidden_size', 'N/A'),
                "num_layers": getattr(self.config, 'num_hidden_layers', getattr(self.config, 'num_layers', 'N/A')),
                "num_attention_heads": getattr(self.config, 'num_attention_heads', 'N/A'),
            })
        
        return info
    
    def save_features(self, features: Dict[str, torch.Tensor], save_path: str) -> None:
        """保存提取的特征"""
        torch.save(features, save_path)
        print(f"✓ 特征已保存到: {save_path}")


# 使用示例函数
def example_usage():
    """使用示例"""
    
    print("=== HuggingFace模型加载和特征提取示例 ===\n")
    
    # 示例1: 从Hub加载BERT模型并提取文本特征
    print("1. 加载BERT模型并提取文本特征")
    try:
        loader = HuggingFaceModelLoader("bert-base-uncased")
        loader.load_model_from_hub("bert-base-uncased")
        
        # 注册钩子提取中间特征
        loader.register_hooks_for_feature_extraction(['encoder.layer.6', 'encoder.layer.11'])
        
        # 提取特征
        text = "Hello, this is a test sentence for feature extraction."
        features = loader.extract_features_text(text)
        
        print(f"提取到 {len(features)} 个特征层")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
        
        # 显示模型信息
        info = loader.get_model_info()
        print(f"模型信息: {info}")
        
    except Exception as e:
        print(f"BERT示例失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 从Hub加载ViT模型并提取图像特征
    print("2. 加载ViT模型并提取图像特征")
    try:
        loader = HuggingFaceModelLoader("google/vit-base-patch16-224")
        loader.load_model_from_hub("google/vit-base-patch16-224")
        
        # 注册钩子
        loader.register_hooks_for_feature_extraction(['encoder.layer.6', 'encoder.layer.11'])
        
        # 创建示例图像
        from PIL import Image
        import numpy as np
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 提取特征
        features = loader.extract_features_image(dummy_image)
        
        print(f"提取到 {len(features)} 个特征层")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
        
    except Exception as e:
        print(f"ViT示例失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例3: 从本地文件加载模型（需要实际的模型文件）
    print("3. 从本地文件加载模型示例")
    print("注意: 需要实际的.bin或.safetensor文件")
    
    # 示例代码（需要实际文件路径）
    """
    loader = HuggingFaceModelLoader("local_model")
    
    # 加载.bin格式
    loader.load_model_from_bin("path/to/model.bin", "path/to/config.json")
    
    # 或加载.safetensor格式
    loader.load_model_from_safetensor("path/to/model.safetensors", "path/to/config.json")
    
    # 注册钩子并提取特征
    loader.register_hooks_for_feature_extraction()
    features = loader.extract_features_text("test text")
    """


if __name__ == "__main__":
    example_usage()