import os
import copy
import json
import torch
import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from utils.augmentation import BatchAugmentation_battery_revised
from data_provider.data_split_recorder import split_recorder
from denseweight import DenseWeight

warnings.filterwarnings('ignore')

datasetName2ids = {
    'CALCE': 0,
    'HNEI': 1,
    'HUST': 2,
    'MATR': 3,
    'RWTH': 4,
    'SNL': 5,
    'MICH': 6,
    'MICH_EXP': 7,
    # 'Tongji1': 8,
    'Tongji': 8,
    'Stanford': 9,
    'ISU-ILCC': 11,
    'XJTU': 12,
    'ZN-coin': 13,
    'UL-PUR': 14,
    # 'Tongji2': 15,
    # 'Tongji3': 16,
    'CALB': 17,
    'ZN42': 22,
    'ZN2024': 23,
    'CALB42': 24,
    'CALB2024': 25,
    'NA-ion': 27,
    'NA-ion42': 28,
    'NA-ion2024': 29,
}
datasetName2batteryType = {
    'CALCE': 0, 'HNEI': 0, 'HUST': 0, 'MATR': 0, 'RWTH': 0, 'SNL': 0, 'MICH': 0, 'MICH_EXP': 0,
    'Tongji1': 0, 'Stanford': 0, 'ISU-ILCC': 0, 'XJTU': 0, 'UL-PUR': 0, 'Tongji2': 0, 'Tongji3': 0,  # Li-ion
    'ZN-coin': 1, 'ZN42': 1, 'ZN2024': 1,  # Zn-ion
    'NA-ion': 2, 'NA-ion42': 2, 'NA-ion2024': 2,  # Na-ion
    'CALB': 3, 'CALB42': 3, 'CALB2024': 3  # CALB
}


def my_collate_fn_baseline(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])

    # Choose which target to use based on prediction task
    # 修正：使用字典键访问方式
    task = samples[0].get('prediction_task', 'multi')

    if task == 'SOC':
        labels = torch.Tensor([i['soc_labels'] for i in samples])
    elif task == 'SOH':
        labels = torch.Tensor([i['soh_labels'] for i in samples])
    elif task == 'multi':
        # Multi-task: only return SOC + SOH
        soc_labels = torch.Tensor([i['soc_labels'] for i in samples])
        soh_labels = torch.Tensor([i['soh_labels'] for i in samples])
        labels = torch.stack([soc_labels, soh_labels], dim=1)
    else:
        raise ValueError(f"Unknown task: {task}")

    weights = torch.Tensor([i['weight'] for i in samples])
    battery_type_ids = torch.LongTensor([i['battery_type_id'] for i in samples])

    return cycle_curve_data, curve_attn_mask, labels, weights, battery_type_ids


class Dataset_original(Dataset):
    def __init__(self, args, flag='train', label_scalers=None, tokenizer=None, eval_cycle_max=None, eval_cycle_min=None,
                 use_target_dataset=False):
        '''
        init the Dataset_BatteryFormer class for SOC/SOH prediction
        :param args: model parameters
        :param flag: including train, val, test
        :param label_scalers: dict of scalers for different targets {'soc': scaler, 'soh': scaler}
        '''
        # 在方法开始处添加
        print(f"Dataset_original init - flag: {flag}")
        print(f"Dataset: {args.dataset}")
        print(f"Data path: {args.root_path}")
        self.eval_cycle_max = eval_cycle_max
        self.eval_cycle_min = eval_cycle_min
        self.args = args
        self.root_path = os.path.join(args.root_path, f"{args.dataset}_dataset")

        self.seq_len = args.seq_len
        self.charge_discharge_len = args.charge_discharge_length
        self.flag = flag
        self.dataset = args.dataset if not use_target_dataset else args.target_dataset
        self.early_cycle_threshold = args.early_cycle_threshold
        self.prediction_task = getattr(args, 'prediction_task', 'multi')  # 'SOC', 'SOH' or 'multi'
        self.KDE_samples = []

        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah',
                          'time_in_s']
        self.aug_helper = BatchAugmentation_battery_revised()
        assert flag in ['train', 'test', 'val', 'adapt']

        # Initialize dataset files based on dataset name
        self._initialize_dataset_files()

        # 在文件加载后添加
        print(f"Raw files found: {len(self.files) if hasattr(self, 'files') else 'No files'}")

        # Read data
        (self.total_charge_discharge_curves, self.total_curve_attn_masks,
         self.total_soc_labels, self.total_soh_labels,
         self.total_dataset_ids, self.total_battery_type_ids) = self.read_data()

        # 修正：确保有数据时才进行KDE样本设置
        if len(self.total_soc_labels) > 0:
            self.KDE_samples = copy.deepcopy(
                self.total_soc_labels + self.total_soh_labels ) if flag == 'train' else []
        else:
            self.KDE_samples = []

        self.weights = self.get_loss_weight()

        if len(self.total_charge_discharge_curves) > 0 and np.any(np.isnan(self.total_charge_discharge_curves)):
            raise Exception('Nan in the data')

            # 在数据过滤后添加
        print(f"Filtered training data: {len(self.train_data) if hasattr(self, 'train_data') else 'No train_data'}")

        # 在抛出异常的地方之前添加
        #if flag == 'train' and len(self.train_data) == 0:
        if flag == 'train' and len(self.total_soc_labels) == 0:
            print("DEBUG: About to raise 'No training data found' exception")
            print(f"Filtered training data: {len(self.total_soc_labels) if hasattr(self, 'total_soc_labels') else 'No training data'}")
            print(f"Other relevant args: {vars(args)}")


        # Initialize label scalers
        if flag == 'train' and label_scalers is None:
            if len(self.total_soc_labels) == 0:
                raise Exception('No training data found')

            self.label_scalers = {
                'soc': StandardScaler(),
                'soh': StandardScaler()
            }

            # Fit scalers on training data
            self.label_scalers['soc'].fit(np.array(self.total_soc_labels).reshape(-1, 1))
            self.label_scalers['soh'].fit(np.array(self.total_soh_labels).reshape(-1, 1))

            # Transform training data
            self.total_soc_labels = self.label_scalers['soc'].transform(
                np.array(self.total_soc_labels).reshape(-1, 1)).flatten()
            self.total_soh_labels = self.label_scalers['soh'].transform(
                np.array(self.total_soh_labels).reshape(-1, 1)).flatten()

        else:
            # validation set or testing set
            assert label_scalers is not None
            self.label_scalers = label_scalers

            # Transform validation/test data using training scalers
            if len(self.total_soc_labels) > 0:
                self.total_soc_labels = self.label_scalers['soc'].transform(
                    np.array(self.total_soc_labels).reshape(-1, 1)).flatten()
                self.total_soh_labels = self.label_scalers['soh'].transform(
                    np.array(self.total_soh_labels).reshape(-1, 1)).flatten()


    def _initialize_dataset_files(self):
        """Initialize train/val/test file lists based on dataset"""
        # few-shot / 自定义 split：允许从 args 注入 file list
        injected = getattr(self.args, 'custom_split_files', None)
        if isinstance(injected, dict) and self.flag in injected and injected[self.flag] is not None:
            self.train_files = injected.get('train', [])
            self.val_files = injected.get('val', [])
            self.test_files = injected.get('test', [])
            self.adapt_files = injected.get('adapt', [])
            self.files = list(injected[self.flag])
            return

        if self.dataset == 'exp':
            self.train_files = split_recorder.Stanford_train_files[:3]
            self.val_files = split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            self.test_files = split_recorder.Tongji_test_files[:2] + split_recorder.HUST_test_files[:2]
        elif self.dataset == 'Tongji':
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == 'HUST':
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == 'MATR':
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == 'SNL':
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == 'MICH':
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == 'MICH_EXP':
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == 'UL_PUR':
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == 'RWTH':
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == 'HNEI':
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == 'CALCE':
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == 'Stanford':
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == 'ISU_ILCC':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'XJTU':
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == 'MIX_large':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_ALL':
            self.train_files = split_recorder.MIX_ALL_train_files
            self.val_files = split_recorder.MIX_ALL_val_files
            self.test_files = split_recorder.MIX_ALL_test_files
        elif self.dataset == 'ZN-coin':
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files
            self.test_files = split_recorder.ZNcoin_test_files
        elif self.dataset == 'CALB':
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == 'ZN-coin42':
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == 'ZN-coin2024':
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == 'CALB42':
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == 'CALB2024':
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == 'NAion':
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == 'NAion42':
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == 'NAion2024':
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        elif self.dataset == 'Test_Liion':
            self.train_files = split_recorder.Test_Liion_train_files
            self.val_files = split_recorder.Test_Liion_val_files
            self.test_files = split_recorder.Test_Liion_test_files
        elif self.dataset == 'Test_NAion':
            self.train_files = split_recorder.Test_NAion_train_files
            self.val_files = split_recorder.Test_NAion_val_files
            self.test_files = split_recorder.Test_NAion_test_files
        elif self.dataset == 'Test_ZNion':
            self.train_files = split_recorder.Test_ZNion_train_files
            self.val_files = split_recorder.Test_ZNion_val_files
            self.test_files = split_recorder.Test_ZNion_test_files
        if self.flag == 'train':
            self.files = list(self.train_files)
        elif self.flag == 'val':
            self.files = list(self.val_files)
        elif self.flag == 'test':
            self.files = list(self.test_files)
        elif self.flag == 'adapt':  # 支持 adapt 阶段
            self.files = list(getattr(self, 'adapt_files', []))

    def get_loss_weight(self, method='KDE'):
        '''
        Get the weight for weighted loss
        method can be ['1/n', '1/log(x+1)', 'KDE']
        '''
        if getattr(self.args, 'weighted_loss', False) and self.flag == 'train' and len(self.total_dataset_ids) > 0:
            if method == '1/n':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()
                weights = 1.0 / label_to_count[df["label"]].values
            elif method == '1/log(x+1)':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()
                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x) + 1)
                weights = 1 / normalized_x
            elif method == 'KDE':
                if len(self.KDE_samples) > 0:
                    # Define DenseWeight
                    dw = DenseWeight(alpha=1.0)
                    # Fit DenseWeight and get the weights for the samples
                    dw.fit(self.KDE_samples)
                    # Calculate the weight for an arbitrary target value
                    weights = []
                    for label in self.KDE_samples:
                        single_sample_weight = dw(label)[0]
                        weights.append(single_sample_weight)
                else:
                    weights = np.ones(self.__len__())
            else:
                raise Exception('Not implemented')
            return weights
        else:
            return np.ones(self.__len__())

    def return_label_scalers(self):
        return self.label_scalers

    def __len__(self):
        if len(self.total_soc_labels) > 0:
            return len(self.total_soc_labels)
        else:
            return 0

    def read_data(self):
        '''
        read all data from files
        :return: charge_discharge_curves, curve_attn_masks, soc_labels, soh_labels, dataset_ids, battery_type_ids
        '''
        total_charge_discharge_curves = []  # 修正：重新添加这个
        total_curve_attn_masks = []
        total_dataset_ids = []
        total_battery_type_ids = []
        total_soc_labels = []
        total_soh_labels = []

        # Windows + accelerate/deepspeed 场景下 tqdm 有时会因 stderr 句柄无效而崩溃（WinError 6），这里自动降级禁用进度条
        try:
            file_iter = tqdm(self.files)
        except OSError:
            file_iter = self.files

        for file_name in file_iter:
            # Determine dataset ID
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_prefix = file_name.split('_')[0]
                dataset_id = datasetName2ids.get(dataset_prefix, 0)
            else:
                dataset_id = datasetName2ids['MICH_EXP']

            # Determine battery type
            dataset_prefix = file_name.split('_')[0]
            battery_type_id = datasetName2batteryType.get(dataset_prefix, 0)

            curves, attn_masks, soc_values, soh_values = self.read_samples_from_one_cell(file_name)

            if curves is None:
                continue

            total_charge_discharge_curves += curves  # 修正：重新添加
            total_curve_attn_masks += attn_masks
            total_soc_labels += soc_values
            total_soh_labels += soh_values


            # Add dataset and battery type IDs for each sample
            num_samples = len(curves)
            total_dataset_ids += [dataset_id] * num_samples
            total_battery_type_ids += [battery_type_id] * num_samples

        return (total_charge_discharge_curves, total_curve_attn_masks,  # 修正：返回格式
                total_soc_labels, total_soh_labels,
                total_dataset_ids, total_battery_type_ids)

    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        prefix = file_name.split('_')[0]
        if prefix.startswith('MATR'):
            data = pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data = pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data = pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data = pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data = pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            if not os.path.isdir(f'{self.root_path}/total_MICH/'):
                self.merge_MICH(f'{self.root_path}/total_MICH/')
            data = pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data = pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data = pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))
        elif prefix.startswith('UL-PUR'):
            data = pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))
        elif prefix.startswith('SMICH'):
            data = pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb'))
        elif prefix.startswith('BIT2'):
            data = pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb'))
        elif prefix.startswith('Tongji'):
            data = pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            data = pickle.load(open(f'{self.root_path}/Stanford/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data = pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data = pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data = pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data = pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data = pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))

        # Load EOL labels
        if prefix == 'MICH':
            with open(f'{self.root_path}/Life_labels/total_MICH_labels.json') as f:
                life_labels = json.load(f)
            lookup_name = file_name
        elif prefix.startswith('Tongji'):
            file_name_corrected = file_name.replace('--', '-#')
            with open(f'{self.root_path}/Life_labels/Tongji_labels.json') as f:
                life_labels = json.load(f)
            lookup_name = file_name_corrected  # ← 关键修复
        else:
            with open(f'{self.root_path}/Life_labels/{prefix}_labels.json') as f:
                life_labels = json.load(f)
            lookup_name = file_name

        # 使用正确的查找名称
        if lookup_name in life_labels:
            eol = life_labels[lookup_name]
        else:
            eol = None

        return data, eol

    def read_cell_df(self, file_name):
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, eol, nominal_capacity, cj_aug_charge_discharge_curves
        '''
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]

        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']

        cycle_data = data['cycle_data']  # list of cycle data dict

        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan  # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah'] < 0] = np.nan
            cycle_df.bfill(inplace=True)  # deal with NaN
            total_cycle_dfs.append(cycle_df)

            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break

        df = pd.concat(total_cycle_dfs)
        # obtain the charge and discharge curves
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold,
                                                                   nominal_capacity)
        cj_aug_charge_discharge_curves, fm_aug_charge_discharge_curves = self.aug_helper.batch_aug(
            charge_discharge_curves)

        return df, charge_discharge_curves, eol, nominal_capacity, cj_aug_charge_discharge_curves

    def read_samples_from_one_cell(self, file_name):
        '''
        read all samples using this function
        :param file_name: which file needs to be read
        :return: curves, attn_masks, soc_values, soh_values
        '''
        print(f"Processing file: {file_name}")
        df, charge_discharge_curves_data, eol, nominal_capacity, cj_aug_charge_discharge_curves = self.read_cell_df(
            file_name)
        if df is None:
            print(f"  -> Skipped: df is None (no EOL or other issue)")
            return None, None, None, None

        print(f"  -> EOL: {eol}, cycles available: {df['cycle_number'].max()}")
        print(f"  -> Processing range: {self.seq_len} to {min(self.early_cycle_threshold + 1, eol)}")

        curves = []
        attn_masks = []
        soc_values = []  # State of Charge
        soh_values = []  # State of Health

        # Get initial capacity for SOH calculation
        initial_capacity = df[df['cycle_number'] == 1]['discharge_capacity_in_Ah'].max()

        # Get data for the early-life cycles
        for i in range(self.seq_len, min(self.early_cycle_threshold + 1, eol)):
            if i >= eol:
                # If we encounter a battery whose cycle life is even smaller than early_cycle_threshold
                # We should not include the eol cycle data
                break
            cycle_data = df[df['cycle_number'] == i]
            if len(cycle_data) > 0:
                # SOC: Current capacity / Nominal capacity
                current_capacity = cycle_data['discharge_capacity_in_Ah'].max()
                soc = current_capacity / nominal_capacity
                # SOH: Current capacity / Initial capacity
                soh = current_capacity / initial_capacity

                soc_values.append(soc)
                soh_values.append(soh)

                # Add charge-discharge curve data
                if i <= len(charge_discharge_curves_data):
                    curves.append(charge_discharge_curves_data[i - 1])

            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1  # set 1 not to mask

            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if i <= self.eval_cycle_max and i >= self.eval_cycle_min:
                    # Only keep the val and test samples that satisfy the eval_cycle
                    pass
                else:
                    continue

            attn_masks.append(tmp_attn_mask)

        print(f"  -> Generated {len(soc_values)} samples")
        return curves, attn_masks, soc_values, soh_values

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []
        unique_cycles = df['cycle_number'].unique()
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle in range(1, early_cycle_threshold + 1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]

                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records / nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values
                time_in_s_records = cycle_df['time_in_s'].values

                cutoff_voltage_indices = np.nonzero(
                    current_records_in_C >= 0.01)  # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][
                    -1]  # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C <= -0.01)
                discharge_end_index = cutoff_voltage_indices[0][-1]

                if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_35', 'CALB_45']:
                    # Every cycle first discharge and then charge
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]

                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity

                    charge_voltages = charge_voltages[np.abs(charge_current_in_C) > 0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C) > 0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C) > 0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C) > 0.01]
                else:
                    # Every cycle first charge and then discharge
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity

                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C) > 0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C) > 0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C) > 0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C) > 0.01]

                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(
                    discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(
                    charge_voltages, charge_currents, charge_capacities)

                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)

                voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(
                    voltage_records)  # normalize using the cutoff voltage
                current_records = current_records.reshape(1,
                                                          self.charge_discharge_len) / nominal_capacity  # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1,
                                                                  self.charge_discharge_len) / nominal_capacity  # normalize the capacity

                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))

        curves = np.concatenate(curves, axis=0)  # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or discharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interpolated records
        '''
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages) + 1)
        interp_bases = np.linspace(1, len(voltages) + 1, num=charge_discharge_len,
                                   endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery

    def __getitem__(self, index):
        sample = {
            'cycle_curve_data': torch.Tensor(self.total_charge_discharge_curves[index]),
            'curve_attn_mask': torch.Tensor(self.total_curve_attn_masks[index]),
            'soc_labels': self.total_soc_labels[index],
            'soh_labels': self.total_soh_labels[index],
            'weight': self.weights[index],
            'dataset_id': self.total_dataset_ids[index],
            'battery_type_id': torch.tensor(self.total_battery_type_ids[index], dtype=torch.long),
            'prediction_task': self.prediction_task
        }
        return sample

    def read_train_labels(self, train_files):
        train_labels = []
        for file_name in train_files:
            prefix = file_name.split('_')[0]
            if prefix == 'MICH':
                with open(f'{self.root_path}/total_MICH_labels.json') as f:
                    life_labels = json.load(f)
            elif prefix.startswith('Tongji'):
                with open(f'{self.root_path}/Tongji_labels.json') as f:
                    life_labels = json.load(f)
            else:
                with open(f'{self.root_path}/{prefix}_labels.json') as f:
                    life_labels = json.load(f)
            if file_name in life_labels:
                eol = life_labels[file_name]
            else:
                continue
            train_labels.append(eol)
        return train_labels

    def merge_MICH(self, merge_path):
        os.makedirs(merge_path)
        source_path1 = f'{self.root_path}/MICH/'
        source_path2 = f'{self.root_path}/MICH_EXP/'
        source1_files = [i for i in os.listdir(source_path1) if i.endswith('.pkl')]
        source2_files = [i for i in os.listdir(source_path2) if i.endswith('.pkl')]
        target_path = f'{self.root_path}/total_MICH/'

        for file in source1_files:
            shutil.copy(source_path1 + file, target_path)
        for file in source2_files:
            shutil.copy(source_path2 + file, target_path)