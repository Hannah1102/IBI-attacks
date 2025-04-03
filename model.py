
import torch
import torch.nn as nn
import pytorch_lightning as pl


class EmbeddingSEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16, is_tanh=False, is_activate=True):
        """
        input_dim: 输入的特征维度，即词嵌入的维度
        reduction: 用于减少特征维度的缩减系数，默认为16
        """
        super(EmbeddingSEBlock, self).__init__()
        # 压缩阶段
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，沿时间维度压缩
        # 激发阶段
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)  # 降维
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)  # 升维
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.is_tanh = is_tanh
        self.is_activate = is_activate

    def forward(self, x):
        """
        x: 输入的文本特征矩阵，形状为 (batch_size, seq_len, embedding_dim)
        """
        # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Squeeze操作: 全局平均池化，得到 (batch_size, embedding_dim, 1)
        se = self.global_avg_pool(x)
        
        # Flatten，形状变为 (batch_size, embedding_dim)
        se = se.view(se.size(0), -1)
        
        # Excitation操作：经过两个全连接层并使用ReLU和Sigmoid
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        if self.is_activate:
            if self.is_tanh:
                se = self.tanh(se)
            else:
                se = self.sigmoid(se)

        # 对输入的每个特征维度进行加权 (batch_size, embedding_dim, seq_len)
        se = se.unsqueeze(-1)  # 扩展维度 (batch_size, embedding_dim, 1)
        # x = x * se  # 逐通道加权 (batch_size, embedding_dim, seq_len)
        
        # 转换回原始形状 (batch_size, seq_len, embedding_dim)
        # x = x.transpose(1, 2)

        # 只返回权重，而不是加权之后的结果，因为加权并不在自身
        se = se.transpose(1, 2)

        return se



class TokenSEBlock(nn.Module):
    def __init__(self, seq_len, reduction=8, is_tanh=False, is_activate=True):
        super(TokenSEBlock, self).__init__()
        # 压缩阶段
        self.global_avg_pool = nn.AdaptiveAvgPool2d((seq_len, 1))  # 全局平均池化，沿时间维度压缩
        # 激发阶段
        self.fc1 = nn.Linear(seq_len, seq_len // reduction, bias=False)  # 降维
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(seq_len // reduction, seq_len, bias=False)  # 升维
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.is_tanh = is_tanh
        self.is_activate = is_activate

    def forward(self, x):
        """
        x: 输入的文本特征矩阵，形状为 (batch_size, seq_len, embedding_dim)
        """
        
        # Squeeze操作: 全局平均池化，得到 (batch_size, seq_len, 1)
        se = self.global_avg_pool(x)
        
        # Flatten，形状变为 (batch_size, seq_len)
        se = se.view(se.size(0), -1)
        
        # Excitation操作：经过两个全连接层并使用ReLU和Sigmoid
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        if self.is_activate:
            if self.is_tanh:
                se = self.tanh(se)
            else:
                se = self.sigmoid(se)

        se = se.unsqueeze(-1)  # 扩展维度 (batch_size, seq_len, 1)

        return se

class AdjustTextVectors(nn.Module):
    def __init__(self, seq_len, embedding_dim, reduction=16, is_tanh=False):
        super(AdjustTextVectors, self).__init__()
        self.embedding_se_block = EmbeddingSEBlock(input_dim=embedding_dim, reduction=16, is_tanh=is_tanh)  # embedding attention
        self.token_se_block = TokenSEBlock(seq_len=seq_len, reduction=8, is_tanh=is_tanh)  # token attention
        self.is_tanh = is_tanh

    def forward(self, diff_vector, normal_vector):
        """
        diff_vector: 平均差值向量，形状为 (batch_size, seq_len, embedding_dim)
        normal_vector: 正常文本向量，形状为 (batch_size, seq_len, embedding_dim)
        """
        # 使用SE模块计算基于正常文本向量的注意力权重
        embedding_attention = self.embedding_se_block(normal_vector)
        # embedding_attention = 1.0 # tanh: 0, sigmoid: 1

        token_attention = self.token_se_block(normal_vector)
        # token_attention = 1.0 # tanh: 0, sigmoid: 1
        
        # 扩展 diff_vector，使第一个维度与 batch_size 一致
        diff_vector = diff_vector.expand(normal_vector.shape[0], -1, -1)  # -1 表示保持该维度不变.在不需要改变数据的时候使用expand，在需要改变的时候使用repeat（占用内存大）


        # 对差值向量应用注意力权重
        # print(f'diff_vector: {diff_vector.shape}, attention_weights: {attention_weights.shape}')
        if self.is_tanh:
            embedding_weighted_diff = diff_vector * (1 + embedding_attention)
            weighted_diff = embedding_weighted_diff * (1 + token_attention)
        else:
            embedding_weighted_diff = diff_vector * embedding_attention
            weighted_diff = embedding_weighted_diff * token_attention

        # 将加权后的差值向量添加到正常文本向量上
        adjusted_vector = normal_vector + weighted_diff

        return adjusted_vector

# use exiting functions
# torchvision.ops.SqueezeExcitation

### pytorch module
# input = torch.randn(50, 512, 7, 7)
# kernel_size = input.shape[2]
# cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
# output = cbam(input)
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, activation=None):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # nn.Conv2d(channel, reduction, 1, bias=False),
            activation(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
            # nn.Conv2d(reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output



class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=4):
        super(SimpleTransformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, input_dim)  # 输出维度等于输入维度

    def forward(self, src):
        # Transformer 输入应该是 [sequence_length, batch_size, input_dim]
        # 需要对输入进行转置
        src = src.permute(1, 0, 2)  # [seq_length, batch_size, input_dim]
        transformer_out = self.transformer(src)
        # 再将结果转置回来 [batch_size, seq_length, input_dim]
        transformer_out = transformer_out.permute(1, 0, 2)
        out = self.fc_out(transformer_out)
        return out

class MLP(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, self.output_size) 
        self.layer_norm1 = nn.LayerNorm(1024)
        self.layer_norm2 = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(0.2)  # Dropout层

    def forward(self, x):
        batch_size = x.size()[0] 
        # print(f'batch_size, num_tokens, feature_dim: {batch_size, num_tokens, feature_dim}')
        x = x.view(batch_size, -1)  # 展平为 [batch_size, 77 * 1024]
        # print(f'x.shape: {x.shape}')  # [4, 78848]
        x = torch.relu(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = x.view(batch_size, 77, 1024) 
        return x