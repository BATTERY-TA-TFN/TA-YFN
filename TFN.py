from torch.nn import Conv1d
import torch.nn.init as init
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

fmin = 0.03
fmax = 0.45
random_scale = 2e-3

class BaseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.phases = ['real', 'imag']
        self.weight = torch.Tensor(len(self.phases), out_channels, in_channels, self.kernel_size)
        if bias:
            self.bias = torch.Tensor(len(self.phases), out_channels)
        else:
            self.bias = None

        for phase in self.phases:
            self.weight[self.phases.index(phase)] = torch.Tensor(out_channels, in_channels, self.kernel_size)

            # --- 原始代码 (Kaiming Initialization) ---
            # init.kaiming_uniform_(self.weight[self.phases.index(phase)], a=math.sqrt(5))  # initial weight

            # --- 修改后的代码 (Xavier Uniform Initialization) ---
            # Xavier Uniform适用于Tanh/Sigmoid，对于一般非线性是一个保守的选择
            init.xavier_uniform_(self.weight[self.phases.index(phase)], gain=1.0)  # gain=1.0 is default for linear/ReLU

            if bias:
                self.bias[self.phases.index(phase)] = torch.Tensor(out_channels)
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[self.phases.index(phase)])  # initial bias
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias[self.phases.index(phase)], -bound, bound)

        if self.__class__.__name__ == 'BaseConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)

    def forward(self, input):
        result = {}
        for phase in self.phases:
            if self.bias is None:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
            else:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         bias=self.bias[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
        output = torch.sqrt(result[self.phases[0]].pow(2) + result[self.phases[1]].pow(2))
        return output

class BaseFuncConv1d(BaseConv1d):
    def __init__(self, *pargs, **kwargs):
        kwargs_new = {}
        for k in kwargs.keys():
            if k in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']:
                kwargs_new[k] = kwargs[k]
        super().__init__(*pargs, **kwargs_new)
        if self.__class__.__name__ == 'BaseFuncConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)
            self.random_weight = self.weight

    def _clamp_parameters(self):
        with torch.no_grad():
            for i in range(len(self.params_bound)):
                self.superparams.data[:, :, i].clamp_(self.params_bound[i][0], self.params_bound[i][1])

    def WeightForward(self):
        if self.clamp_flag:
            self._clamp_parameters()
        l00 = []
        for phase in self.phases:
            l0 = []
            for i in range(self.superparams.shape[0]):
                l1 = [self.weightforward(self.kernel_size, self.superparams[i, j], phase) for j in
                      range(self.superparams.shape[1])]
                l0.append(torch.stack(l1, dim=0).unsqueeze(0))
            l00.append(torch.cat(l0, dim=0).unsqueeze(0))
            self.weight = torch.cat(l00, dim=0)

    def forward(self, input):
        if self.__class__.__name__ != 'BaseFuncConv1d':
            self.WeightForward()
        return super().forward(input)



class TFconv_STTF(BaseFuncConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 clamp_flag=True):
        params_bound = ((1e-3, 0.5), (0.05, 2.0)) # <--- 边界放宽

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.superparams = nn.Parameter(torch.cat([
            torch.rand(out_channels, in_channels, 1) * 0.4 + 0.05,  # freq (0.05~0.45)
            torch.ones(out_channels, in_channels, 1) * 0.52  # sigma (初始化为原值 0.52)
        ], dim=-1))

        self._reset_parameters()
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            temp0 = torch.linspace(fmin, fmax, shape.numel()).reshape(shape)
            self.superparams.data[:, :, 0] = temp0
            self.WeightForward()

    def weightforward(self, lens, params, phase):
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        T = torch.arange(-(lens // 2), lens - (lens // 2), device=params.device)
        freq = torch.clamp(params[0], 1e-3, 0.5)
        sigma = torch.clamp(params[1], 0.05, 2.0)  # <--- clamp 范围放宽

        base = torch.exp(-(T / (lens // 2 + 1e-6)).pow(2) / (sigma.pow(2) * 2 + 1e-6))

        if self.phases.index(phase) == 0:
            result = base * torch.cos(2 * math.pi * freq * T)
        else:
            result = base * torch.sin(2 * math.pi * freq * T)
        epsilon = 1e-6
        result = result / (torch.norm(result, p=2) + epsilon)

        return torch.nan_to_num(result, nan=0.0, posinf=1e-6, neginf=-1e-6)


class TFconv_STTF_TaskAdaptive(TFconv_STTF):
    """
    针对SOC和SOH分别设计混合策略
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)

        self.task_type = task_type  # 'soc' or 'soh'

        # 🔧 时域分支
        self.time_branch = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )

        if task_type == 'soc':
            # SOC: 需要时域瞬态 → 初始alpha=0.6 (偏向频域但保留时域)
            self.alpha = nn.Parameter(torch.tensor(0.6))
        else:  # 'soh'
            # SOH: 需要平滑趋势 → 初始alpha=0.9 (强偏向频域，抑制时域噪声)
            self.alpha = nn.Parameter(torch.tensor(0.9))
            # 🔧 为SOH添加低通滤波器
            self.soh_lowpass = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, input):
        # 频域输出
        freq_output = super().forward(input)

        # 时域输出
        time_output = self.time_branch(input)

        if self.task_type == 'soh':
            # SOH专用：对时域分支应用低通滤波
            time_output = self.soh_lowpass(time_output)

        # 简单线性混合（移除gate机制）
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        output = alpha_clamped * freq_output + (1 - alpha_clamped) * time_output

        return output


if __name__ == '__main__':
    for item in [TFconv_STTF]:
        print(item.__name__)
        model = TFconv_STTF(1, 8, 15, padding=7, stride=1,
                            bias=False, clamp_flag=True)  # kernel_size should
        input = torch.randn([1, 1, 1024])
        out = model(input)
        out.mean().backward()
        print(out.shape)
        print(f'{item.__name__:s} test pass')

def get_random_features(d_model, num_features, ortho_groups=1, device=None):
    seed = np.random.randint(0, 100000)
    def qr_mv(shape, seed):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        mat = torch.randn(shape, generator=gen, device=device)
        q, _ = torch.linalg.qr(mat)
        return q.transpose(0, 1) if shape[0] < shape[1] else q
    features = qr_mv((num_features, d_model), seed)
    features /= math.sqrt(num_features)

    return features  # 形状: [128, 16]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.2):
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # 使用 GELU 替代 ReLU，Transformer中更常用
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class StandardMultiheadAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        H = self.heads

        # 1. QKV 投影并分割 (3 * dim -> 3 * dim/heads * heads)
        qkv = self.to_qkv(x).reshape(B, L, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q, k, v shape: [B, H, L, head_dim]

        # 2. 计算 Attention 矩阵: Q * K.T
        # [B, H, L, head_dim] x [B, H, head_dim, L] -> [B, H, L, L]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Softmax 归一化 (标准 Attention 的核心区别)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 4. Attention 加权 V: Attn x V
        # [B, H, L, L] x [B, H, L, head_dim] -> [B, H, L, head_dim]
        out = torch.matmul(attn, v)

        # 5. 重组 Head 并输出
        # [B, H, L, head_dim] -> [B, L, H, head_dim] -> [B, L, C]
        out = out.transpose(1, 2).reshape(B, L, C)

        return self.to_out(out)


class StandardTransformerEncoderLayer(nn.Module):
    # 替换 PerformerEncoderLayer
    def __init__(self, d_model, heads=4, dropout=0.1, d_ff=None):
        super().__init__()
        self.d_model = d_model

        # 核心替换：使用标准 MHA
        self.attn = StandardMultiheadAttention(
            dim=d_model,
            heads=heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)  # FeedForward 保持不变
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x 形状: [B, D, L] (从 CNN 出来)
        x_permuted = x.permute(0, 2, 1)  # [B, L, D] (Transformer 输入)

        identity = x_permuted
        # Attention: (LayerNorm + Add + Attention)
        x_attn = self.norm1(identity + self.attn(identity))

        identity = x_attn
        # FFN: (LayerNorm + Add + FFN)
        x_out = self.norm2(identity + self.ffn(identity))

        return x_out.permute(0, 2, 1)  # [B, D, L] (CNN 输出)

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, norm_groups=4):
        super().__init__()
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(norm_groups, out_channels), out_channels)
            )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(norm_groups, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(norm_groups, out_channels), out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class CNN(nn.Module):
    def __init__(self, mid_channel=16, seq_len=100, num_attn_layers=2):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BasicBlock1D(mid_channel, 16, stride=1, norm_groups=4),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(16, 32, stride=1, norm_groups=8),  # 32 channels, 8 groups (原为 4)
            nn.MaxPool1d(2)
        )
        self.conv3 = BasicBlock1D(32, 64, stride=1, norm_groups=16)  # 64 channels, 16 groups (原为 8)

        attn_layers = []
        d_model = 64
        for _ in range(num_attn_layers):
            attn_layers.append(
                StandardTransformerEncoderLayer(  # ⬅️ ✅ 替换为标准 Transformer Layer
                    d_model=d_model,
                    heads=4,
                    dropout=0.2,  # 标准 Transformer 常用 0.1，但这里保持原值 0.2
                    d_ff=d_model * 2
                )
            )
        self.attn_stack = nn.Sequential(*attn_layers)
        self.maxpool3 = nn.MaxPool1d(2)

        self.layer4_soc = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4)
        )
        self.layer4_soh = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4)
        )
        self.final_feature_dim = 128 * 4
        # 优化后的 layer5_soc 结构：
        self.layer5_soc = nn.Sequential(
            nn.Linear(self.final_feature_dim, 512),  # 512 -> 512 (增大宽度)
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),  # 512 -> 256 (增加深度)
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),  # 256 -> 64 (保持输出维度)
            nn.ReLU(inplace=True)
        )

        # 优化后的 layer5_soh 结构：
        self.layer5_soh = nn.Sequential(
            nn.Linear(self.final_feature_dim, 512),  # 512 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),  # 256 -> 64
            nn.ReLU(inplace=True)
        )
        self.output_dim = 64  # SOC和SOH特征维度保持一致

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        x = self.layer1(x)
        x_share = self.layer2(x)  # x_share: [B, 32, L/4]
        x_conv3 = self.conv3(x_share)  # [B, 64, L/4]

        # --- 关键修改开始 ---
        # 1. 计算共享的 Attention 特征 (只计算一次)
        # 注意：attn_stack 的输入和输出是 [B, C, L] 格式
        # 但 PerformerEncoderLayer 内部操作需要 [B, L, C]
        # x_conv3.permute(0, 2, 1) -> [B, L/4, 64]
        x_attn_output = self.attn_stack(x_conv3)  # attn_stack 的输出是 [B, 64, L/4] (内部已处理转置)

        # 2. SOC 路径：使用 Attention 输出
        x_soc = self.maxpool3(x_attn_output)  # [B, 64, L/8]

        # 3. SOH 路径：也使用 Attention 输出
        # SOH 依赖长期退化信息，Attention 是关键
        x_soh = self.maxpool3(x_attn_output)  # [B, 64, L/8]

        x_soc = self.layer4_soc(x_soc)
        x_soc = x_soc.reshape(x_soc.size(0), -1)
        soc_output = self.layer5_soc(x_soc)  # [B, 64]
        x_soh = self.layer4_soh(x_soh)
        x_soh = x_soh.reshape(x_soh.size(0), -1)
        soh_output = self.layer5_soh(x_soh)  # [B, 64]

        return soc_output, soh_output

class Base_FUNC_CNN(nn.Module):
    FuncConv1d = None
    funckernel_size = 21

    def __init__(self, in_channels=1, mid_channel=16, seq_len=100, clamp_flag=True):
        super().__init__()
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim  # 64

        # TFN 第一层
        self.funconv = self.FuncConv1d(
            in_channels, mid_channel, self.funckernel_size,
            padding=self.funckernel_size // 2,
            # bias=False, <--- 移除或保留为 False 都可以, 因为 TFconv_* 已经设置为 True
            clamp_flag=clamp_flag
        )
        self.superparams = self.funconv.superparams

    def forward(self, x):
        x = self.funconv(x)  # x: [B, mid_channel, S]
        soc_features, soh_features = self.cnn_backbone(x)
        return soc_features, soh_features

    def getweight(self):
        weight = self.funconv.weight.cpu().detach().numpy()
        superparams = self.funconv.superparams.cpu().detach().numpy()
        return weight, superparams


class TFN_STTF(Base_FUNC_CNN):
    FuncConv1d = TFconv_STTF

    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1  # e.g., 16*8-1 = 127
        super().__init__(mid_channel=mid_channel, seq_len=seq_len, **kwargs)

# 在 TFN_STTF 定义后添加这个类
class TFN_STTF_TaskAdaptive(nn.Module):
    """
    任务自适应的TFN模型
    - SOC路径：使用alpha=0.6的混合策略
    - SOH路径：使用alpha=0.9的混合策略 + 低通滤波
    """

    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        super().__init__()
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim  # 64

        funckernel_size = mid_channel * 8 - 1
        clamp_flag = kwargs.get('clamp_flag', True)

        # 🔧 为SOC和SOH分别创建不同的funconv
        self.funconv_soc = TFconv_STTF_TaskAdaptive(
            in_channels=1,
            out_channels=mid_channel,
            kernel_size=funckernel_size,
            padding=funckernel_size // 2,
            clamp_flag=clamp_flag,
            task_type='soc'
        )

        self.funconv_soh = TFconv_STTF_TaskAdaptive(
            in_channels=1,
            out_channels=mid_channel,
            kernel_size=funckernel_size,
            padding=funckernel_size // 2,
            clamp_flag=clamp_flag,
            task_type='soh'
        )

        # 用于兼容性
        self.superparams = None

    def forward(self, x):
        # x: [B*C, 1, S]
        # SOC路径
        x_soc = self.funconv_soc(x)
        soc_features, _ = self.cnn_backbone(x_soc)

        # SOH路径
        x_soh = self.funconv_soh(x)
        _, soh_features = self.cnn_backbone(x_soh)

        return soc_features, soh_features

    def getweight(self):
        """返回SOC路径的权重（用于可视化）"""
        weight_soc = self.funconv_soc.weight.cpu().detach().numpy()
        superparams_soc = self.funconv_soc.superparams.cpu().detach().numpy()
        return weight_soc, superparams_soc


class Random_conv(Conv1d):
    def __init__(self, *pargs, **kwargs):
        new_kwargs = {k: v for k, v in kwargs.items() if
                      k in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']}
        super().__init__(*pargs, **new_kwargs)
        self.random_weight = self.weight


class Random_CNN(Base_FUNC_CNN):
    FuncConv1d = Random_conv

    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)


class Model(nn.Module):
    def __init__(self, configs, tfn_model_type=TFN_STTF_TaskAdaptive):
        super(Model, self).__init__()
        self.configs = configs
        if configs.c_out == 2:
            self.c_out_soc = 1
            self.c_out_soh = 1
        elif configs.c_out == 1:
            raise ValueError("configs.c_out must represent total output dimension for SOC and SOH.")
        else:
            self.c_out_soc = configs.c_out // 2
            self.c_out_soh = configs.c_out - self.c_out_soc

        self.intra_TFN = tfn_model_type(mid_channel=configs.mid_channel)

        if hasattr(self.intra_TFN, 'superparams'):
            self.superparams = self.intra_TFN.superparams
        else:
            self.superparams = None

        self.d_model = self.intra_TFN.final_feature_dim  # 64

        self.projection_soc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, self.c_out_soc)
        )
        self.projection_soh = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, self.c_out_soh)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x_enc, *args, **kwargs):
        B, C, S = x_enc.shape
        intra_input = x_enc.reshape(-1, 1, S)  # [B*C, 1, S]

        def safe_check_tensor(name, x):
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️ NaN/Inf detected after {name}")
                x = torch.nan_to_num(x, nan=1e-6, posinf=1e-6, neginf=-1e-6)
            return x

        def safe_forward(name, layer, x):
            try:
                x = layer(x)
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
                raise
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️ NaN/Inf detected after {name}")
                print(f"   stats -> mean={x.mean().item():.6f}, "
                      f"std={x.std().item():.6f}, "
                      f"min={x.min().item():.6f}, "
                      f"max={x.max().item():.6f}")
                x = torch.nan_to_num(x, nan=1e-6, posinf=1e-6, neginf=-1e-6)
            return x

        try:
            soc_output, soh_output = self.intra_TFN(intra_input)
        except Exception as e:
            print(f"❌ Error in intra_TFN: {e}")
            raise

        soc_output = safe_check_tensor("intra_TFN_SOC", soc_output)
        soh_output = safe_check_tensor("intra_TFN_SOH", soh_output)

        soc_cycle_features = soc_output.reshape(B, C, self.d_model)  # [B, C, 64]
        soc_pooled_output = soc_cycle_features.permute(0, 2, 1)  # [B, 64, C]
        soc_pooled_output = F.adaptive_avg_pool1d(soc_pooled_output, 1).squeeze(-1)

        soh_cycle_features = soh_output.reshape(B, C, self.d_model)  # [B, C, 64]
        soh_pooled_output = soh_cycle_features.permute(0, 2, 1)  # [B, 64, C]
        soh_pooled_output = F.adaptive_avg_pool1d(soh_pooled_output, 1).squeeze(-1)

        preds_soc = safe_forward("projection_soc", self.projection_soc, soc_pooled_output)
        preds_soh = safe_forward("projection_soh", self.projection_soh, soh_pooled_output)

        preds = torch.cat([preds_soc, preds_soh], dim=-1)

        return preds