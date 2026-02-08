import os
import time
import json
import torch
import wandb
import random
import joblib
import socket
import argparse
import datetime
import accelerate
import numpy as np
from copy import deepcopy

from torch import nn, optim
from torch.optim import lr_scheduler
from visualization import BatteryPredictionVisualizer
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from battery_evaluation import vali_comprehensive_with_warning, BatteryEvaluationAndWarning
from data_provider.data_factory import data_provider_baseline
from utils.tools import get_parameter_number, del_files, EarlyStopping, adjust_learning_rate
from models import CPGRU, CPLSTM, CPMLP, CPBiGRU, CPBiLSTM, CPTransformer, PatchTST, iTransformer, Transformer, \
    DLinear, Autoformer, MLP, MICN, CNN, \
    BiLSTM, BiGRU, GRU, LSTM, \
    TimeMixer, TimesNet, TimesNet_MLP, Timesformer, CPMixer, TFN, TFNablation, WTconv, TimeXer, TFN_YUAN, MSGNet, TimeFilter, LiPM, Chronos, TimeMoE, CPTimeMoE

# 导入消融实验模型
try:
    from models import TFN_ablation
except ImportError:
    TFN_ablation = None
    print("Warning: TFN_ablation module not found. Ablation experiments will not be available.")


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser(description='BatteryLife')

# 修改后代码
def set_seed(seed):
    # 优先尝试使用 accelerate 的设置（若不存在也不抛错）
    try:
        accelerate.utils.set_seed(seed)
    except Exception:
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 为可复现设置，但尽量不要完全禁用 cuDNN（除非你确实需要）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # PyTorch 1.8+ 推荐启用 deterministic 算法（若运行时不支持，会抛异常）
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except Exception:
        # 如果你的环境或某些 op 不支持 deterministic 算法，可以忽略或记录
        pass


# 计算均方根误差 (RMSE)
def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def convert_numpy_types(obj):
    """Recursively converts numpy types in a dictionary to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


# basic config
parser.add_argument('--task_name', type=str, required=False, default='battery_prediction',
                    help='task name for battery SOC/SOH prediction')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='CPTimeMoE',
                    help='model name, options: [Autoformer, DLinear, CPTransformer, TimeMixer, TimesNet, TimesNet_MLP, Timesformer, CPMixer, TFN, TFN_ablation, TFNablation, WTconv]')
parser.add_argument('--seed', type=int, default=2024, help='random seed')

# 预测任务类型选择
parser.add_argument('--device', type=str,  default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--prediction_task', type=str, default='multi',
                    choices=['SOC', 'SOH','multi'],
                    help='Prediction task: SOC, SOH, or multi-task')

# data loader
parser.add_argument('--charge_discharge_length', type=int, default=100,
                    help='The resampled le0ngth for charge and discharge curves')
parser.add_argument('--dataset', type=str, default='Test_NAion', help='dataset  used for pretrained model, '
                                                                     'options: [CALB2024, MIX_large, NAion2024, ZN-coin, MIX_ALL, Test_Liion, Test_NAion, Test_ZNion]')
parser.add_argument('--data', type =str, required=False, default='BatteryLife', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:h urly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='when to stop model training')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--lstm_layers', type=int, default=1, help='num of LSTM layers')
parser.add_argument('--e_layers', type=int, default=2, help='num of intra-cycle layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of inter-cycle layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=10, help='patch length')
parser.add_argument('--stride', type=int, default=10, help='stride')
parser.add_argument('--patch_len2', type=int, default=10, help='patch length for inter-cycle patching')
parser.add_argument('--stride2', type=int, default=10, help='stride for inter-cycle patching')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')

# TimeXer
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layer s')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')

# optimization
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--weighted_sampling', action='store_true', default=False, help='use weighted sampling')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=160, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5,
                    help='The model is trained at least some epochs before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=30, help='e arly stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss func  tion, [MSE, BMSE, MAPE]')

parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, default=8, help='gradient accumulation steps')#TFN:8
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--num_battery_types', type=int, default=4,
                    help='number of battery types (Li-ion=0, Zn-ion=1, Na-ion=2, CALB=3)')
parser.add_argument('--batt_embed_dim', type=int, default=1, help='battery type embedding dimension')

parser.add_argument('--mid_channel', type=int, default=16, help='hidden channels for TFN first layer')
# 在 run_main.py 中添加参数
parser.add_argument('--inter_layers', type=int, default=2, help='number of inter-cycle transformer layers')
#
# # 消融实验参数
# parser.add_argument('--ablation_type', type=str, default='full',
#                                     choices=['full',
#                                              'soc_only_true',  # 【新增】真正的SOC单任务
#                                              'soh_only_true',  # 【新增】真正的SOH单任务
#                                              'macro_M1_off_single_path',
#                                              'macro_M2_off_fixed_stft',
#                                              'macro_M3_transformer_only',
#                                              'macro_M3_cnn_only',
#                                              'no_task_adaptive','no_time_branch','fixed_alpha','no_lowpass','fixed_freq','standard_cnn',
#                                              'shared_tfconv','shared_alpha','swap_alpha','freq_only','time_only','freeze_f','freeze_sigma',
#                                              'tie_f','tie_sigma'],
#                                     help='TA-TFN 消融类型（含宏消融：macro_* 与兼容小粒度）')
parser.add_argument('--fixed_alpha_soc', type=float, default=0.3, help='固定的 α_soc')
parser.add_argument('--fixed_alpha_soh', type=float, default=0.9, help='固定的 α_soh')

parser.add_argument('--fixed_alpha', type=float, default=0.5, help='Fixed alpha value for ablation experiment')
parser.add_argument('--fixed_freq', type=float, default=0.1, help='Fixed frequency value for ablation experiment')

#FNet
parser.add_argument('--fnet_d_ff', type=int, default=256, help='FFN dimension for FNet block')
parser.add_argument('--fnet_d_model', type=int, default=64, help='embedding size for FNet block')
parser.add_argument('--complex_dropout', type=float, default=0.1, help='dropout rate for complex block')
parser.add_argument('--fnet_layers', type=int, default=2, help='number of FNet layers')

# Explainability switches
parser.add_argument('--explain', action='store_true', default=True,
                    help='输出 TFconv 频谱图、O-FR、时域/频域/时频图')
parser.add_argument('--explain_fs', type=float, default=None,
                    help='采样频率(Hz)。不提供则用归一化频率(0~0.5)')
parser.add_argument('--explain_bands', type=int, default=32,
                    help='O-FR 频带数')
parser.add_argument('--explain_task_index', type=int, default=None,
                    help='O-FR 只看多任务输出中的某一维(默认 None 为 L2 范数)')

parser.add_argument('--adapt_ratio', type=float, default=0.05, help='few-shot adapt ratio (0~1), 支持例如 0.01/0.03/0.05')
parser.add_argument('--adapt_epochs', type=int, default=5, help='finetune epochs for few-shot adapt stage')
parser.add_argument('--adapt_lr', type=float, default=1e-4, help='learning rate for few-shot adapt stage')
parser.add_argument('--sample_ratio', type=float, default=1, help='subsample ratio for train/val/test file lists (0~1)')
parser.add_argument('--sample_seed', type=int, default=42, help='random seed for subsampling')
parser.add_argument('--few_shot_seed', type=int, default=42, help='shuffle seed before slicing adapt set')

args = parser.parse_args()
# =========================================
# few-shot split 计算（仅当选择跨化学测试数据集时生效）
# =========================================
from data_provider.data_split_recorder import split_recorder as _sr

def build_custom_split(ds_name:str, ratio:float, seed:int, max_resample:int=50, sample_ratio:float=1.0, sample_seed:int=42):
    """根据数据集名称生成 custom_split_files dict。

    注意：部分文件可能因缺少 EOL 标签而在 Dataset 中被跳过（df is None），
    这里对 adapt 文件做“可用性探测 + 重采样”，避免 adapt 数据集为空。
    """
    rng = np.random.RandomState(seed)
    if ds_name == 'Test_NAion':
        target_pool = list(_sr.NAion_2024_train_files)
        adapt_dir = os.path.join(args.root_path, f"{args.dataset}_dataset", 'NA-ion')
    elif ds_name == 'Test_Liion':
        target_pool = list(_sr.CALB_2024_train_files)
        adapt_dir = os.path.join(args.root_path, f"{args.dataset}_dataset", 'CALB')
    elif ds_name == 'Test_ZNion':
        target_pool = list(_sr.ZNcoin_train_files)
        adapt_dir = os.path.join(args.root_path, f"{args.dataset}_dataset", 'ZN-coin')
    else:
        return None  # 非跨化学 few-shot 数据集

    k = max(1, int(len(target_pool) * ratio))

    # 轻量可用性检查：如果 life label json 里没有该 key，Dataset 会直接跳过
    def is_likely_usable(fn: str) -> bool:
        prefix = fn.split('_')[0]
        label_path = os.path.join(args.root_path, f"{args.dataset}_dataset", 'Life_labels', f"{prefix}_labels.json")
        try:
            with open(label_path, 'r') as f:
                labels = json.load(f)
            return fn in labels
        except Exception:
            return True

    # 先过滤一次，尽量用“有标签”的文件
    filtered_pool = [f for f in target_pool if is_likely_usable(f)]
    if len(filtered_pool) >= k:
        target_pool = filtered_pool

    adapt_files = None
    for _ in range(max_resample):
        rng.shuffle(target_pool)
        cand = target_pool[:k]
        # 至少保证文件存在（更进一步的样本生成在 Dataset 内部做）
        if all(os.path.exists(os.path.join(adapt_dir, x)) for x in cand):
            adapt_files = cand
            break
    if adapt_files is None:
        rng.shuffle(target_pool)
        adapt_files = target_pool[:k]

    # 保留原 split 作为 train/val/test
    train_attr = f"{ds_name}_train_files"
    val_attr = f"{ds_name}_val_files"
    test_attr = f"{ds_name}_test_files"
    source_train = list(getattr(_sr, train_attr))
    source_val = list(getattr(_sr, val_attr))
    source_test = list(getattr(_sr, test_attr))

    def subsample_list(lst, ratio, seed):
        if ratio is None or ratio >= 1.0:
            return list(lst)
        rng2 = np.random.RandomState(seed)
        n = max(1, int(len(lst) * ratio))
        idx = rng2.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in idx]

    # 对 train/val/test 子采样（减少训练量），adapt 保持独立
    source_train = subsample_list(source_train, sample_ratio, sample_seed)
    source_val = subsample_list(source_val, sample_ratio, sample_seed + 1)
    source_test = subsample_list(source_test, sample_ratio, sample_seed + 2)

    return {
        'train': source_train,  # 预训练阶段
        'val': source_val,
        'test': source_test,
        'adapt': adapt_files,  # few-shot 微调阶段
        'subsample': {
            'ratio': sample_ratio,
            'seed': sample_seed
        }
    }

custom_split = build_custom_split(args.dataset, args.adapt_ratio, args.few_shot_seed, sample_ratio=args.sample_ratio, sample_seed=args.sample_seed)
setattr(args, 'custom_split_files', custom_split)


# Set output dimensions based on task
if args.prediction_task == 'multi':
    args.output_num = 2  # SOC, SOH
    args.c_out = 2
else:
    args.output_num = 1  # Single task
    args.c_out = 1

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        # 复制模型参数
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + (1.0 - self.decay) * msd[k].detach().to(v.device))
                else:
                    v.copy_(msd[k].detach())


def get_final_predictions(args, accelerator, model, test_data, test_loader, label_scalers):
    """获取测试集的最终预测结果用于可视化"""
    model.eval()
    all_predictions = []
    all_actuals = []
    all_cycles = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = batch

            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)
            battery_type_id = battery_type_id.to(accelerator.device)

            try:
                outputs = model(cycle_curve_data, battery_type_id, curve_attn_mask)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
            except Exception as e:
                accelerator.print(f"获取预测结果时失败: {e}")
                continue

            cut_off = labels.shape[0]
            outputs = outputs[:cut_off]

            # 修改后（推荐）
            # 把 outputs/labels 转为 numpy（在 CPU）再做反归一化，避免在 GPU tensor 上就地修改
            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            if args.prediction_task == 'multi':
                denorm_preds = np.zeros_like(outputs_np)
                denorm_labels = np.zeros_like(labels_np)
                for idx, task in enumerate(['soc', 'soh']):
                    task_scaler = label_scalers[task]
                    # 如果 scaler 有 inverse_transform，优先使用它
                    if hasattr(task_scaler, "inverse_transform"):
                        denorm_preds[:, idx] = task_scaler.inverse_transform(outputs_np[:, idx:idx + 1]).reshape(-1)
                        denorm_labels[:, idx] = task_scaler.inverse_transform(labels_np[:, idx:idx + 1]).reshape(-1)
                    else:
                        std = np.sqrt(task_scaler.var_[-1])
                        mean_value = task_scaler.mean_[-1]
                        denorm_preds[:, idx] = outputs_np[:, idx] * std + mean_value
                        denorm_labels[:, idx] = labels_np[:, idx] * std + mean_value
            else:
                task_scaler = label_scalers[args.prediction_task.lower()]
                if hasattr(task_scaler, "inverse_transform"):
                    denorm_preds = task_scaler.inverse_transform(outputs_np.reshape(-1, 1)).reshape(-1)
                    denorm_labels = task_scaler.inverse_transform(labels_np.reshape(-1, 1)).reshape(-1)
                else:
                    std = np.sqrt(task_scaler.var_[-1])
                    mean_value = task_scaler.mean_[-1]
                    denorm_preds = outputs_np * std + mean_value
                    denorm_labels = labels_np * std + mean_value

            all_predictions.append(denorm_preds)
            all_actuals.append(denorm_labels)

            # 生成循环编号
            batch_size = len(outputs)
            batch_cycles = list(range(len(all_cycles), len(all_cycles) + batch_size))
            all_cycles.extend(batch_cycles)

    return {
        'predictions': np.concatenate(all_predictions, axis=0),
        'actuals': np.concatenate(all_actuals, axis=0),
        'cycles': np.array(all_cycles)
    }


def main():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2_baseline.json')
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision='fp16'
    )
    accelerator.print(args.__dict__)

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_lradj{}_dataset{}_loss{}_wd{}_wl{}_bs{}_s{}'.format(
            args.model,
            args.seq_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.lradj, args.dataset, args.loss, args.wd, args.weighted_loss, args.batch_size, args.seed)

        data_provider_func = data_provider_baseline

        # 模型初始化代码保持不变...
        if args.model == 'Transformer':
            model = Transformer.Model(args).float()
        elif args.model == 'CPBiLSTM':
            model = CPBiLSTM.Model(args).float()
        elif args.model == 'CPBiGRU':
            model = CPBiGRU.Model(args).float()
        elif args.model == 'CPGRU':
            model = CPGRU.Model(args).float()
        elif args.model == 'CPLSTM':
            model = CPLSTM.Model(args).float()
        elif args.model == 'BiLSTM':
            model = BiLSTM.Model(args).float()
        elif args.model == 'CPTimeMoE':
            model = CPTimeMoE.Model(args).float()
        elif args.model == 'BiGRU':
            model = BiGRU.Model(args).float()
        elif args.model == 'LSTM':
            model = LSTM.Model(args).float()
        elif args.model == 'GRU':
            model = GRU.Model(args).float()
        elif args.model == 'PatchTST':
            model = PatchTST.Model(args).float()
        elif args.model == 'iTransformer':
            model = iTransformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        elif args.model == 'CPMLP':
            model = CPMLP.Model(args).float()
        elif args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'MLP':
            model = MLP.Model(args).float()
        elif args.model == 'MICN':
            model = MICN.Model(args).float()
        elif args.model == 'CNN':
            model = CNN.Model(args).float()
        elif args.model == 'CPTransformer':
            model = CPTransformer.Model(args).float()
        elif args.model == 'TimeMixer':
            model = TimeMixer.Model(args).float()
        elif args.model == 'TimesNet':
            model = TimesNet.Model(args).float()
        elif args.model == 'TimesNet_MLP':
            model = TimesNet_MLP.Model(args).float()
        elif args.model == 'Timesformer':
            model = Timesformer.Model(args).float()
        elif args.model == 'CPMixer':
            model = CPMixer.Model(args).float()
        elif args.model == 'TFN':
            model = TFN.Model(args).float()
        elif args.model == 'MSGNet':
            model = MSGNet.Model(args).float()
        elif args.model == 'TimeFilter':
            model = TimeFilter.Model(args).float()
        elif args.model == 'TFNablation':
            model = TFNablation.Model(args).float()
        elif args.model == 'WTconv':
            model = WTconv.Model(args).float()
        elif args.model == 'TimeXer':
            model = TimeXer.Model(args).float()
        elif args.model == 'LiPM':
            model = LiPM.Model(args).float()
        elif args.model == 'TFN_YUAN':
            model = TFN_YUAN.Model(args).float()
        elif args.model == 'Chronos':
            model = Chronos.Model(args).float()
        elif args.model == 'TFN_ablation':
            if TFN_ablation is None:
                raise ImportError("TFN_ablation module is not available. Please check the import.")
            ablation_type = getattr(args, 'ablation_type', 'full')
            model = TFN_ablation.Model(args, ablation_type=ablation_type).float()
        else:
            raise Exception(f'The {args.model} is not an implemented baseline!')
        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)

        accelerator.print("Loading training samples......")
        # 修正：使用正确的参数名和方法名
        train_data, train_loader = data_provider_func(args, 'train', label_scalers=None)
        label_scalers = train_data.return_label_scalers()  # 修正：使用复数形式
        accelerator.print("Loading vali samples......")
        vali_data, vali_loader = data_provider_func(args, 'val', label_scalers=label_scalers)  # 修正：传递正确参数
        accelerator.print("Loading test samples......")
        test_data, test_loader = data_provider_func(args, 'test', label_scalers=label_scalers)  # 修正：传递正确参数

        if accelerator.is_local_main_process and os.path.exists(path):
            del_files(path)
            accelerator.print(f'success delete {path}')

        os.makedirs(path, exist_ok=True)
        # 只有在分布式训练时才执行同步
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

        joblib.dump(label_scalers, f'{path}/label_scalers')  # 修正：保存scalers字典

        with open(path + '/args.json', 'w') as f:
            json.dump(args.__dict__, f)

        if getattr(args, 'custom_split_files', None) is not None:
            with open(os.path.join(path, 'split_used.json'), 'w') as f:
                json.dump(args.custom_split_files, f)

        def can_connect(host="api.wandb.ai", port=443, timeout=3):
            try:
                socket.setdefaulttimeout(timeout)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                return True
            except Exception:
                return False

        # W&B 初始化
        use_wandb = False
        if accelerator.is_local_main_process:
            if can_connect():
                try:
                    wandb.init(
                        project=f"Battery_{args.prediction_task}_Prediction",
                        config=args.__dict__,
                        name=f"{args.prediction_task}_{nowtime}",
                        settings=wandb.Settings(init_timeout=120)
                    )
                    use_wandb = True
                except Exception as e:
                    accelerator.print(f"W&B 初始化失败: {e}")
                    use_wandb = False
                    os.environ["WANDB_MODE"] = "disabled"
            else:
                os.environ["WANDB_MODE"] = "disabled"
                accelerator.print("无法连接 W&B，已禁用 wandb 日志上传")

        para_res = get_parameter_number(model)
        accelerator.print(para_res)

        for name, module in model._modules.items():
            accelerator.print(name, " : ", module)

        time_now = time.time()
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        trained_parameters_names = []
        for name, p in model.named_parameters():
            if p.requires_grad is True:
                trained_parameters_names.append(name)
                trained_parameters.append(p)

        accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
        model_optim = optim.AdamW(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)

        # 先 prepare model & optimizer & dataloaders（不提前创建依赖于 train_steps 的 scheduler）
        train_loader, vali_loader, test_loader, model, model_optim = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim
        )

        # 现在计算 train_steps（准备完之后，DistributedSampler 的长度已被正确设置）
        train_steps = len(train_loader)

        # 根据 train_steps 创建 scheduler
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.MSELoss(reduction='none')

        best_vali_loss = float('inf')
        best_vali_MAE, best_test_MAE = 0, 0
        best_vali_RMSE, best_test_RMSE = 0, 0
        best_vali_MAPE, best_test_MAPE = 0, 0

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        evaluation_system = BatteryEvaluationAndWarning(
            save_path=f'./results/{args.dataset}_{args.model}_{args.prediction_task}_{timestamp}/evaluation'
        )
        # 添加训练历史记录变量
        train_loss_history = []
        val_loss_history = []
        lr_history = []

        ema = ModelEMA(model, decay=0.999, device=accelerator.device)

        for epoch in range(args.train_epochs):
            iter_count = 0
            total_loss = 0
            total_preds = []
            total_references = []

            model.train()
            epoch_time = time.time()

            # 修正：获取正确的scaler用于反归一化
            if args.prediction_task == 'multi':
                # 多任务情况下，使用平均值或选择主要任务的scaler
                primary_scaler = label_scalers['soc']  # 或者根据需要选择
            else:
                primary_scaler = label_scalers[args.prediction_task.lower()]

            std = np.sqrt(primary_scaler.var_[-1])
            mean_value = primary_scaler.mean_[-1]

            for i, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    model_optim.zero_grad()
                    iter_count += 1
                    # 修正：正确的数据解包方式
                    cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = batch
                    # print(f"Batch {i}: Cycle Data shape received from DataLoader: {cycle_curve_data.shape}")
                    labels = labels.float().to(accelerator.device)
                    cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                    weights = weights.float().to(accelerator.device)
                    battery_type_id = battery_type_id.to(accelerator.device)

                    # 新增：数据验证，跳过包含NaN的批次
                    if (torch.isnan(cycle_curve_data).any() or torch.isnan(labels).any() or
                            torch.isnan(weights).any() or torch.isinf(cycle_curve_data).any() or
                            torch.isinf(labels).any() or torch.isinf(weights).any()):
                        accelerator.print("Warning: Skipping batch due to NaN/Inf in input data")
                        continue

                    try:
                        outputs = model(cycle_curve_data, battery_type_id, curve_attn_mask)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                    except Exception as e:
                        accelerator.print(f"模型前向传播失败: {e}")
                        continue

                    # 新增：验证模型输出
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        accelerator.print("Warning: Model output contains NaN/Inf, skipping batch")
                        continue

                    cut_off = labels.shape[0]

                    if args.loss == 'MSE':
                        loss = criterion(outputs[:cut_off], labels)

                        if torch.isnan(loss).any():
                            accelerator.print("Warning: MSE loss contains NaN, replacing with zeros")
                            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

                        if args.prediction_task == 'multi':
                            loss = torch.mean(loss, dim=1)
                            loss = torch.mean(loss * weights)
                        else:
                            # 【新增】单任务：squeeze掉最后一维
                            loss = torch.mean(loss.squeeze(-1) * weights)

                    elif args.loss == 'MAPE':
                        epsilon = 1e-6  # 增加epsilon值

                        if args.prediction_task == 'multi':
                            # 对每个任务分别计算MAPE然后平均
                            mape_losses = []
                            for task_idx, task in enumerate(['soc', 'soh']):
                                task_scaler = label_scalers[task]
                                task_std = np.sqrt(task_scaler.var_[-1])
                                task_mean = task_scaler.mean_[-1]

                                tmp_outputs = outputs[:cut_off, task_idx] * task_std + task_mean
                                tmp_labels = labels[:, task_idx] * task_std + task_mean

                                # 修复1: 使用绝对值作为分母，避免负数问题
                                denominator = torch.abs(tmp_labels) + epsilon

                                # 修复2: 计算百分比误差
                                percentage_error = torch.abs((tmp_outputs - tmp_labels) / denominator)

                                # 修复3: 处理NaN和Inf值
                                percentage_error = torch.where(
                                    torch.isnan(percentage_error) | torch.isinf(percentage_error),
                                    torch.zeros_like(percentage_error),
                                    percentage_error
                                )

                                # 修复4: 限制极大值，防止梯度爆炸
                                percentage_error = torch.clamp(percentage_error, min=0.0, max=5.0)

                                mape_losses.append(percentage_error)

                            # 确保weights没有异常值
                            safe_weights = torch.where(
                                torch.isnan(weights) | torch.isinf(weights),
                                torch.ones_like(weights),
                                weights
                            )

                            # 堆叠损失并计算加权平均
                            stacked_losses = torch.stack(mape_losses, dim=0)  # [3, batch_size]
                            weighted_losses = stacked_losses * safe_weights.unsqueeze(0)
                            loss = torch.mean(weighted_losses)

                        else:
                            tmp_outputs = outputs[:cut_off] * std + mean_value
                            tmp_labels = labels * std + mean_value

                            # 修复1: 使用绝对值作为分母
                            denominator = torch.abs(tmp_labels) + epsilon
                            percentage_error = torch.abs((tmp_outputs - tmp_labels) / denominator)

                            # 修复2: 处理NaN和Inf值
                            percentage_error = torch.where(
                                torch.isnan(percentage_error) | torch.isinf(percentage_error),
                                torch.zeros_like(percentage_error),
                                percentage_error
                            )

                            # 修复3: 限制极大值
                            percentage_error = torch.clamp(percentage_error, min=0.0, max=5.0)

                            # 确保weights没有异常值
                            safe_weights = torch.where(
                                torch.isnan(weights) | torch.isinf(weights),
                                torch.ones_like(weights),
                                weights
                            )

                            loss = torch.mean(percentage_error * safe_weights)
                    else:
                        raise ValueError(f"Unknown loss: {args.loss}")

                    # 新增：最终检查loss是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print(f"Warning: Invalid loss detected ({loss.item()}), skipping batch")
                        continue

                    accelerator.backward(loss)
                    model_optim.step()
                    ema.update(model)
                    total_loss += loss.item()

                    # 【修复】反归一化处理 - 单任务需要flatten
                    if args.prediction_task == 'multi':
                        # 多任务：outputs是[B, 2]，使用第一个任务监控
                        transformed_preds = outputs[:cut_off, 0] * std + mean_value
                        transformed_labels = labels[:, 0] * std + mean_value
                    else:
                        # 单任务：outputs是[B, 1]，需要squeeze成[B]
                        transformed_preds = (outputs[:cut_off].squeeze(-1) * std + mean_value)
                        transformed_labels = (labels.squeeze(-1) * std + mean_value)

                    # 新增：清理预测和标签中的异常值
                    valid_mask = ~(torch.isnan(transformed_preds) | torch.isnan(transformed_labels) |
                                   torch.isinf(transformed_preds) | torch.isinf(transformed_labels))

                    if valid_mask.any():
                        clean_preds = transformed_preds[valid_mask]
                        clean_labels = transformed_labels[valid_mask]

                        all_predictions, all_targets = accelerator.gather_for_metrics(
                            (clean_preds, clean_labels))
                        total_preds += all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                        total_references += all_targets.detach().cpu().numpy().reshape(-1).tolist()
                    else:
                        accelerator.print("Warning: All predictions/labels are invalid, skipping metrics update")

                    if args.lradj == 'TST':
                        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

                    if (i + 1) % 10 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        iter_count = 0
                        time_now = time.time()
            # 计算训练指标
            ema.update(model)

            train_loss = total_loss / max(1, iter_count)
            train_rmse = root_mean_squared_error(total_references, total_preds)
            train_mape = mean_absolute_percentage_error(total_references, total_preds)
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 新的代码：
            vali_results = vali_comprehensive_with_warning(
                args, accelerator, model, vali_data, vali_loader,
                criterion, label_scalers, evaluation_system
            )
            test_results = vali_comprehensive_with_warning(
                args, accelerator, model, test_data, test_loader,
                criterion, label_scalers, evaluation_system
            )
            # 从结果中提取传统指标以保持兼容性
            if args.prediction_task == 'multi':
                vali_mae_loss = vali_results['overall_mae']
                vali_rmse = vali_results['overall_rmse']
                vali_mape = vali_results['overall_mape']
                test_mae_loss = test_results['overall_mae']
                test_rmse = test_results['overall_rmse']
                test_mape = test_results['overall_mape']
            else:
                vali_mae_loss = vali_results['mae']
                vali_rmse = vali_results['rmse']
                vali_mape = vali_results['mape']
                test_mae_loss = test_results['mae']
                test_rmse = test_results['rmse']
                test_mape = test_results['mape']

            vali_loss = vali_mae_loss

            # 记录训练历史
            train_loss_history.append(train_loss)
            val_loss_history.append(vali_loss)
            lr_history.append(model_optim.param_groups[0]['lr'])

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_vali_MAE = vali_mae_loss
                best_test_MAE = test_mae_loss
                best_vali_RMSE = vali_rmse
                best_test_RMSE = test_rmse
                best_vali_MAPE = vali_mape
                best_test_MAPE = test_mape
                # >>> 保存最佳模型的详细评估结果 <<<
                best_vali_results = vali_results
                best_test_results = test_results

                if accelerator.is_main_process:
                    try:
                        # 尝试保存模型，如果失败则使用替代方法
                        model_path = os.path.join(path, "best_model.pth")
                        torch.save(model.state_dict(), model_path)
                        accelerator.print(f"Model saved successfully to {model_path}")
                    except RuntimeError as e:
                        accelerator.print(f"Warning: Failed to save model with torch.save: {e}")
                        accelerator.print("Attempting alternative save method...")
                        try:
                            # 尝试使用accelerator的保存方法
                            accelerator.save_model(model, path)
                            accelerator.print(f"Model saved using accelerator.save_model to {path}")
                        except Exception as e2:
                            accelerator.print(f"Error: Both save methods failed. Last error: {e2}")
                            accelerator.print("Model will not be saved. Please check disk space and file permissions.")
                    except Exception as e:
                        accelerator.print(f"Error saving model: {e}")
                        accelerator.print("Please check disk space and file permissions.")

            accelerator.print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} | "
                f"Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | "
                f"Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
                f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")

            if accelerator.is_local_main_process and use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_rmse": train_rmse,
                    "train_mape": train_mape,
                    "vali_RMSE": vali_rmse,
                    "vali_MAPE": vali_mape,
                    "test_RMSE": test_rmse,
                    "test_MAPE": test_mape,
                    "learning_rate": model_optim.param_groups[0]['lr']
                })

            # 修正：EarlyStopping调用参数
            early_stopping(epoch + 1, vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                accelerator.set_trigger()

            if accelerator.check_trigger():
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # =============================
        # Stage 2: few-shot adaptation（仅当存在 custom_split_files['adapt'] 时启用）
        # =============================
        do_few_shot = isinstance(getattr(args, 'custom_split_files', None), dict) and len(args.custom_split_files.get('adapt', [])) > 0
        if do_few_shot:
            accelerator.print(f"[Few-shot] adapt files: {len(args.custom_split_files.get('adapt', []))}, ratio={args.adapt_ratio}, seed={args.few_shot_seed}")

            accelerator.print("[Few-shot] Loading adapt samples......")
            try:
                adapt_data, adapt_loader = data_provider_func(args, 'adapt', label_scalers=label_scalers)
            except Exception as e:
                import traceback
                accelerator.print(f"[Few-shot] Failed to build adapt dataloader: {e}")
                accelerator.print(traceback.format_exc())
                raise

            # 使用独立的优化器进行微调（小 lr、少 epoch）
            adapt_parameters = []
            for _, p in model.named_parameters():
                if p.requires_grad:
                    adapt_parameters.append(p)
            adapt_optim = optim.AdamW(adapt_parameters, lr=args.adapt_lr, weight_decay=args.wd)

            # 只 prepare 新的 loader + optim（model 已经是 prepare 后的 wrapper）
            adapt_loader, adapt_optim = accelerator.prepare(adapt_loader, adapt_optim)

            for aep in range(args.adapt_epochs):
                model.train()
                adapt_loss_sum = 0.0
                adapt_steps = 0
                for i, batch in enumerate(adapt_loader):
                    with accelerator.accumulate(model):
                        adapt_optim.zero_grad()
                        cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = batch
                        labels = labels.float().to(accelerator.device)
                        cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                        weights = weights.float().to(accelerator.device)
                        battery_type_id = battery_type_id.to(accelerator.device)

                        outputs = model(cycle_curve_data, battery_type_id, curve_attn_mask)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]

                        cut_off = labels.shape[0]
                        loss = criterion(outputs[:cut_off], labels)
                        if args.prediction_task == 'multi':
                            loss = torch.mean(loss, dim=1)
                            loss = torch.mean(loss * weights)
                        else:
                            loss = torch.mean(loss.squeeze(-1) * weights)

                        accelerator.backward(loss)
                        adapt_optim.step()

                        adapt_loss_sum += loss.item()
                        adapt_steps += 1

                mean_adapt_loss = adapt_loss_sum / max(1, adapt_steps)
                accelerator.print(f"[Few-shot] Adapt Epoch {aep + 1}/{args.adapt_epochs} | loss={mean_adapt_loss:.6f}")

            # 微调后重新在 test 上评估一次（不改你的 best_model 逻辑，只额外输出）
            try:
                test_results_after_adapt = vali_comprehensive_with_warning(
                    args, accelerator, model, test_data, test_loader,
                    criterion, label_scalers, evaluation_system
                )
                if accelerator.is_local_main_process:
                    with open(os.path.join(path, 'test_after_adapt.json'), 'w') as f:
                        json.dump(convert_numpy_types(test_results_after_adapt), f, ensure_ascii=False, indent=2)
            except Exception as e:
                accelerator.print(f"[Few-shot] evaluate after adapt failed: {e}")

            # 保存微调后的模型权重
            if accelerator.is_main_process:
                try:
                    torch.save(model.state_dict(), os.path.join(path, 'model_after_adapt.pth'))
                except Exception as e:
                    accelerator.print(f"[Few-shot] save model_after_adapt failed: {e}")

        # === 解释性可视化：频谱图 / O-FR / 时域-频域-时频谱 ===
        if args.explain and accelerator.is_local_main_process and hasattr(model, 'explain'):
            try:
                # 取测试集第一个 batch
                sample_batch = next(iter(test_loader))
                cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = sample_batch
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)

                # 只传一个样本，避免生成过多图；可改为 [:N]
                x_enc = cycle_curve_data[:1]  # [B, C, S]

                # 输出目录
                explain_dir = f'./fig_exports/tfn_explain/{args.dataset}_{args.model}'
                # fs: 若不知道采样频率可以先 None；知道的话传 Hz（如 1.0）
                model.explain(
                    x_enc,
                    save_dir=explain_dir,
                    fs=args.explain_fs,  # 例如 1.0
                    ofr_bands=args.explain_bands,  # 例如 32
                    ofr_task_index=args.explain_task_index  # 例如 0 或 1；None 则 L2 范数
                )
                accelerator.print(f'[Explain] 图已保存到: {explain_dir}')
            except Exception as e:
                accelerator.print(f'[Explain] 生成失败: {e}')

         # 训练结束后进行可视化
        if accelerator.is_local_main_process:
            accelerator.print("开始生成可视化结果...")
            # 加载最佳模型
            accelerator.print("Loading best model for final evaluation...")

            state_dict = torch.load(os.path.join(path, "best_model.pth"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)#还会改模型结构（比如再加参数），就把 strict 改成 False
            # strict=True 确保结构完全一致

            # 获取最终测试结果用于可视化
            try:
                final_test_results = get_final_predictions(args, accelerator, model, test_data, test_loader,
                                                   label_scalers)
                if args.prediction_task == 'multi':
                    test_metrics = {
                        'soc_mae': mean_absolute_error(final_test_results['actuals'][:, 0], final_test_results['predictions'][:, 0]),
                        'soh_mae': mean_absolute_error(final_test_results['actuals'][:, 1], final_test_results['predictions'][:, 1]),
                    }
                else:
                    task_mae = mean_absolute_error(final_test_results['actuals'].flatten(), final_test_results['predictions'].flatten())
                    test_metrics = {
                        f"{args.prediction_task.lower()}_mae": task_mae
                    }

                # 调用新添加的方法来设置阈值
                evaluation_system.set_warning_thresholds(test_metrics)

                # 然后调用你的报告生成方法
                evaluation_system.generate_warning_report(final_test_results)

                # 创建可视化器
                visualizer = BatteryPredictionVisualizer(save_path=f'./results/{args.dataset}_{args.model}_{args.prediction_task}_{timestamp}/plots')

                # 1. 预测结果可视化
                if args.prediction_task == 'multi':
                    # 多任务结果可视化
                    predictions_dict = {
                        'SOC': final_test_results['predictions'][:, 0],
                        'SOH': final_test_results['predictions'][:, 1],
                    }
                    actuals_dict = {
                        'SOC': final_test_results['actuals'][:, 0],
                        'SOH': final_test_results['actuals'][:, 1],
                    }
                    visualizer.plot_multi_task_results(predictions_dict, actuals_dict,
                                                       cycles=final_test_results['cycles'])

                    # 单独为每个任务生成对比图
                    for task_idx, task_name in enumerate(['SOC', 'SOH']):
                        visualizer.plot_prediction_comparison(
                            predictions=final_test_results['predictions'][:, task_idx],
                            actuals=final_test_results['actuals'][:, task_idx],
                            task_name=task_name,
                            cycles=final_test_results['cycles'],
                            save_name=f'{task_name}_prediction.png'
                        )
                else:
                    # 单任务结果可视化
                    visualizer.plot_prediction_comparison(
                        predictions=final_test_results['predictions'].flatten(),
                        actuals=final_test_results['actuals'].flatten(),
                        task_name=args.prediction_task,
                        cycles=final_test_results['cycles'],
                        save_name=f'{args.prediction_task}_prediction.png'
                    )

                # 2. 训练历史可视化
                training_history = {
                    'train_loss': train_loss_history,
                    'val_loss': val_loss_history,
                    'learning_rate': lr_history
                }
                visualizer.plot_training_history(training_history)

                accelerator.print("可视化结果已保存到 './results/plots' 目录")


            except Exception as e:
                accelerator.print(f"可视化生成失败: {e}")

    accelerator.print("=" * 80)
    accelerator.print(f"{args.prediction_task} 预测最终结果")
    accelerator.print("=" * 80)
    accelerator.print(
        f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | '
        f'Test MAPE: {best_test_MAPE:.4f} | Val MAE: {best_vali_MAE:.4f} | '
        f'Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f}'
    )
    # >>> 新增：详细评估报告 <<<
    if accelerator.is_local_main_process:
        accelerator.print("\n" + "=" * 80)
        accelerator.print("详细评估和预警分析报告")
        accelerator.print("=" * 80)

        # 打印详细的验证集结果
        print_detailed_results("验证集", best_vali_results, args.prediction_task)

        # 打印详细的测试集结果
        print_detailed_results("测试集", best_test_results, args.prediction_task)

        # 生成预警报告
        if 'threshold_warnings' in best_test_results:
            warning_report = evaluation_system.generate_warning_report(best_test_results)
            accelerator.print("\n预警分析报告:")
            accelerator.print(warning_report)

    # W&B结束
    if accelerator.is_local_main_process and use_wandb:
        final_metrics = {
            "best_vali_RMSE": best_vali_RMSE,
            "best_vali_MAE": best_vali_MAE,
            "best_vali_MAPE": best_vali_MAPE,
            "best_test_RMSE": best_test_RMSE,
            "best_test_MAE": best_test_MAE,
            "best_test_MAPE": best_test_MAPE,
            "final_epoch": epoch + 1,
        }
        wandb.log(final_metrics)
        wandb.finish()


def print_detailed_results(dataset_name, results, prediction_task):
    """打印详细的评估结果"""
    print(f"\n{dataset_name}详细评估结果:")
    print("-" * 50)

    if prediction_task == 'multi':
        # 多任务详细结果
        for task in ['soc', 'soh']:
            print(f"\n{task.upper()}任务:")
            print(f"  MAE: {results[f'{task}_mae']:.4f}")
            print(f"  RMSE: {results[f'{task}_rmse']:.4f}")
            print(f"  MAPE: {results[f'{task}_mape']:.2f}%")
            print(f"  R²: {results[f'{task}_r2']:.3f}")
            print(f"  平均偏差: {results[f'{task}_bias_mean']:.4f}")
            print(f"  偏差标准差: {results[f'{task}_bias_std']:.4f}")

        print(f"\n整体平均指标:")
        print(f"  Overall MAE: {results['overall_mae']:.4f}")
        print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
        print(f"  Overall MAPE: {results['overall_mape']:.2f}%")

    else:
        # 单任务详细结果
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAPE: {results['mape']:.2f}%")
        print(f"  R²: {results['r2']:.3f}")
        print(f"  平均偏差: {results['bias_mean']:.4f}")
        print(f"  偏差标准差: {results['bias_std']:.4f}")

    # 预警信息简要总结
    if 'threshold_warnings' in results:
        print(f"\n预警状态总结:")
        for task, warnings in results['threshold_warnings'].items():
            critical_pct = warnings['critical_percentage']
            warning_pct = warnings['warning_percentage']
            if critical_pct > 0:
                print(f"  {task.upper()}: ⚠️ 严重预警 {critical_pct:.1f}%, 一般预警 {warning_pct:.1f}%")
            elif warning_pct > 0:
                print(f"  {task.upper()}: ⚠️ 一般预警 {warning_pct:.1f}%")
            else:
                print(f"  {task.upper()}: ✅ 正常")


def extract_wandb_metrics(test_results, vali_results, prediction_task):
    """提取用于W&B的详细指标"""
    metrics = {}

    # 基础指标
    if prediction_task == 'multi':
        for task in ['soc', 'soh']:
            metrics[f'final_test_{task}_mae'] = test_results[f'{task}_mae']
            metrics[f'final_test_{task}_rmse'] = test_results[f'{task}_rmse']
            metrics[f'final_test_{task}_mape'] = test_results[f'{task}_mape']
            metrics[f'final_test_{task}_r2'] = test_results[f'{task}_r2']

            metrics[f'final_vali_{task}_mae'] = vali_results[f'{task}_mae']
            metrics[f'final_vali_{task}_rmse'] = vali_results[f'{task}_rmse']
            metrics[f'final_vali_{task}_mape'] = vali_results[f'{task}_mape']
            metrics[f'final_vali_{task}_r2'] = vali_results[f'{task}_r2']

        metrics['final_test_overall_mae'] = test_results['overall_mae']
        metrics['final_test_overall_rmse'] = test_results['overall_rmse']
        metrics['final_test_overall_mape'] = test_results['overall_mape']

    else:
        task = prediction_task.lower()
        metrics[f'final_test_{task}_mae'] = test_results['mae']
        metrics[f'final_test_{task}_rmse'] = test_results['rmse']
        metrics[f'final_test_{task}_mape'] = test_results['mape']
        metrics[f'final_test_{task}_r2'] = test_results['r2']

    # 预警指标
    if 'threshold_warnings' in test_results:
        for task, warnings in test_results['threshold_warnings'].items():
            metrics[f'warning_{task}_critical_pct'] = warnings['critical_percentage']
            metrics[f'warning_{task}_warning_pct'] = warnings['warning_percentage']

    # 置信度指标
    if 'confidence_analysis' in test_results:
        conf = test_results['confidence_analysis']
        metrics['confidence_mean'] = conf['mean_confidence']
        metrics['confidence_low_pct'] = conf['low_confidence_percentage']

    return metrics

if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()
    main()