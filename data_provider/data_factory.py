from data_provider.data_loader import Dataset_original
from data_provider.data_loader import my_collate_fn_baseline
from torch.utils.data import DataLoader, RandomSampler, Dataset

data_dict = {
    'Dataset_original': Dataset_original,
    'BatteryLife': Dataset_original  # 为了向后兼容保留
}


def data_provider_baseline_DA(args, flag, tokenizer=None, label_scalers=None, eval_cycle_min=None, eval_cycle_max=None,
                              sample_weighted=False, target_dataset='None'):
    """
    Domain Adaptation版本的数据提供器，适配SOC/SOH预测
    """
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # 创建主数据集
    data_set = Data(args=args,
                    flag=flag,
                    label_scalers=label_scalers,
                    eval_cycle_min=eval_cycle_min,
                    eval_cycle_max=eval_cycle_max
                    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=my_collate_fn_baseline)

    if target_dataset != 'None' and flag == 'train':
        target_data_set = Data(args=args,
                               flag=flag,
                               label_scalers=data_set.return_label_scalers(),  # 使用训练集的scalers
                               eval_cycle_min=eval_cycle_min,
                               eval_cycle_max=eval_cycle_max,
                               use_target_dataset=True
                               )

        target_data_loader = DataLoader(
            target_data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=my_collate_fn_baseline)

        target_sampler = RandomSampler(target_data_loader.dataset, replacement=True,
                                       num_samples=len(data_loader.dataset))
        target_resampled_dataloader = DataLoader(target_data_loader.dataset, batch_size=batch_size,
                                                 sampler=target_sampler, collate_fn=my_collate_fn_baseline)
        return data_set, data_loader, target_data_set, target_resampled_dataloader
    else:
        return data_set, data_loader


def data_provider_baseline(args, flag, tokenizer=None, label_scalers=None, eval_cycle_min=None, eval_cycle_max=None,
                           sample_weighted=False):
    """
    标准版本的数据提供器，适配SOC/SOH预测
    支持 few-shot：flag 可为 'train'/'val'/'test'/'adapt'
    """
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
                    flag=flag,
                    label_scalers=label_scalers,
                    eval_cycle_min=eval_cycle_min,
                    eval_cycle_max=eval_cycle_max
                    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=my_collate_fn_baseline)

    return data_set, data_loader


def data_provider_evaluate(args, flag, tokenizer=None, label_scalers=None, eval_cycle_min=None, eval_cycle_max=None,
                           sample_weighted=False):
    """
    评估版本的数据提供器，适配SOC/SOH 预测
    """
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
                    flag=flag,
                    label_scalers=label_scalers,
                    eval_cycle_min=eval_cycle_min,
                    eval_cycle_max=eval_cycle_max
                    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=my_collate_fn_baseline)  # 修正：使用正确的collate函数

    return data_set, data_loader