import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import torch
import os

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def vali_comprehensive_with_warning(args, accelerator, model, vali_data, vali_loader,
                                    criterion, label_scalers, evaluation_system=None):
    """
    增强的验证函数：支持多任务评价和预警分析
    """
    model.eval()
    # ----------------  推理计时初始化 ----------------
    total_infer_time = 0.0  # 秒
    total_infer_samples = 0
    total_loss = []

    # 为每个任务单独存储预测和真实值
    task_preds = {'soc': [], 'soh': []}
    task_trues = {'soc': [], 'soh': []}
    confidence_scores = []
    cycle_numbers = []
    battery_types = []

    # 添加数据加载器检查
    if len(vali_loader) == 0:
        print("⚠️ 警告：验证数据加载器为空，返回默认结果")
        if args.prediction_task == 'multi':
            return {
                'overall_mae': float('inf'),
                'overall_rmse': float('inf'),
                'overall_mape': float('inf'),
                'soc_mae': float('inf'), 'soc_rmse': float('inf'), 'soc_mape': float('inf'), 'soc_r2': 0.0,
                'soh_mae': float('inf'), 'soh_rmse': float('inf'), 'soh_mape': float('inf'), 'soh_r2': 0.0,
                'soc_bias_mean': 0.0, 'soc_bias_std': 0.0,
                'soh_bias_mean': 0.0, 'soh_bias_std': 0.0,
            }
        else:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'bias_mean': 0.0,
                'bias_std': 0.0,
            }

    with torch.no_grad():
        for i, batch in enumerate(vali_loader):
            if batch is None:
                print(f"⚠️ 警告：第{i}个batch为None，跳过")
                continue
            cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = batch

            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)
            weights = weights.float().to(accelerator.device)
            battery_type_id = battery_type_id.to(accelerator.device)

            # ---------------- 单 batch 计时 ----------------
            start_t = time.perf_counter()
            try:
                outputs = model(cycle_curve_data, battery_type_id, curve_attn_mask)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            except Exception as e:
                batch_time = time.perf_counter() - start_t
                total_infer_time += batch_time
                total_infer_samples += cycle_curve_data.size(0)
                accelerator.print(f"验证时模型调用失败: {e}")
                continue
            batch_time = time.perf_counter() - start_t
            total_infer_time += batch_time
            total_infer_samples += cycle_curve_data.size(0)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            cut_off = labels.shape[0]
            outputs = outputs[:cut_off]

            # 计算损失
            loss = criterion(outputs, labels)
            if args.prediction_task == 'multi':
                loss = torch.mean(loss, dim=1)
                loss = torch.mean(loss * weights)
            else:
                loss = torch.mean(loss * weights)
            total_loss.append(loss.detach().cpu())

            # 计算置信度分数（基于预测方差）
            if args.prediction_task == 'multi':
                pred_variance = torch.var(outputs, dim=1).mean().detach().cpu().numpy()
            else:
                pred_variance = torch.var(outputs).detach().cpu().numpy()
            confidence = 1 / (1 + pred_variance)  # 简单的置信度计算
            confidence_scores.extend([confidence] * cut_off)

            # 存储电池类型和循环数（如果可用）
            battery_types.extend(battery_type_id.detach().cpu().numpy())
            cycle_numbers.extend(list(range(len(cycle_numbers), len(cycle_numbers) + cut_off)))

            # 分别处理每个任务的反归一化
            if args.prediction_task == 'multi':
                task_names = ['soc', 'soh']
                for idx, task in enumerate(task_names):
                    task_scaler = label_scalers[task]
                    std = float(np.nan_to_num(np.sqrt(task_scaler.var_[-1]), nan=1.0, posinf=1.0, neginf=1.0))
                    if std < 1e-6:
                        std = 1e-6
                    mean_value = task_scaler.mean_[-1]

                    pred = outputs[:, idx].detach().cpu().numpy() * std + mean_value
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    true = np.nan_to_num(labels[:, idx].detach().cpu().numpy() * std + mean_value,
                                         nan=0.0, posinf=0.0, neginf=0.0)
                    true = labels[:, idx].detach().cpu().numpy() * std + mean_value

                    task_preds[task].append(pred)
                    task_trues[task].append(true)
            else:
                task_name = args.prediction_task.lower()
                task_scaler = label_scalers[task_name]
                std = np.sqrt(task_scaler.var_[-1])
                mean_value = task_scaler.mean_[-1]

                pred = outputs.detach().cpu().numpy() * std + mean_value
                true = labels.detach().cpu().numpy() * std + mean_value

                task_preds[task_name].append(pred)
                task_trues[task_name].append(true)

    # 计算评价指标和预警分析
    results = {}
    total_loss = np.average(total_loss)

    if args.prediction_task == 'multi':
        task_names = ['soc', 'soh']
        overall_mae, overall_rmse, overall_mape = [], [], []

        # 合并所有预测结果
        all_predictions = {}
        all_actuals = {}

        for task in task_names:
            preds = np.concatenate(task_preds[task], axis=0)
            trues = np.concatenate(task_trues[task], axis=0)

            # 确保是一维数组并且类型正确
            preds = np.array(preds, dtype=np.float64).flatten()
            trues = np.array(trues, dtype=np.float64).flatten()

            # 基础指标计算
            mae = mean_absolute_error(trues, preds)
            mse = mean_squared_error(trues, preds)
            rmse = np.sqrt(mse)
            epsilon = 1e-8
            mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + epsilon))) * 100

            # 高级评估指标 - 安全计算 R²
            if len(trues) > 1 and len(preds) > 1 and np.std(trues) > 1e-10 and np.std(preds) > 1e-10:
                try:
                    r2 = stats.pearsonr(trues, preds)[0] ** 2
                except Exception as e:
                    print(f"Warning: Failed to calculate R² for {task}, setting to 0. Error: {e}")
                    r2 = 0.0
            else:
                print(f"Warning: Insufficient variance in {task} data for R² calculation, setting to 0")
                r2 = 0.0

            mean_bias = np.mean(preds - trues)
            std_bias = np.std(preds - trues)
            mean_bias = np.mean(preds - trues)
            std_bias = np.std(preds - trues)

            results[f'{task}_mae'] = mae
            results[f'{task}_rmse'] = rmse
            results[f'{task}_mape'] = mape
            results[f'{task}_r2'] = r2
            results[f'{task}_bias_mean'] = mean_bias
            results[f'{task}_bias_std'] = std_bias

            all_predictions[task] = preds
            all_actuals[task] = trues

            overall_mae.append(mae)
            overall_rmse.append(rmse)
            overall_mape.append(mape)

            accelerator.print(f"{task.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                              f"MAPE: {mape:.2f}%, R²: {r2:.3f}")

        # 整体平均指标
        results['overall_mae'] = np.mean(overall_mae)
        results['overall_rmse'] = np.mean(overall_rmse)
        results['overall_mape'] = np.mean(overall_mape)

        # 预警分析
        if evaluation_system:
            warning_results = evaluation_system.comprehensive_warning_analysis(
                all_predictions, all_actuals, confidence_scores,
                cycle_numbers, battery_types, args.prediction_task
            )
            results.update(warning_results)

    else:
        # 单任务处理
        task_name = args.prediction_task.lower()
        preds = np.concatenate(task_preds[task_name], axis=0)
        trues = np.concatenate(task_trues[task_name], axis=0)

        # 确保是一维数组并且类型正确
        preds = np.array(preds, dtype=np.float64).flatten()
        trues = np.array(trues, dtype=np.float64).flatten()

        mae = mean_absolute_error(trues, preds)
        mse = mean_squared_error(trues, preds)
        rmse = np.sqrt(mse)
        epsilon = 1e-8
        mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + epsilon))) * 100

        # 安全计算 R²
        if len(trues) > 1 and len(preds) > 1 and np.std(trues) > 1e-10 and np.std(preds) > 1e-10:
            try:
                r2 = stats.pearsonr(trues, preds)[0] ** 2
            except Exception as e:
                print(f"Warning: Failed to calculate R², setting to 0. Error: {e}")
                r2 = 0.0
        else:
            print("Warning: Insufficient variance in data for R² calculation, setting to 0")
            r2 = 0.0
        mean_bias = np.mean(preds - trues)
        std_bias = np.std(preds - trues)

        results['mae'] = mae
        results['rmse'] = rmse
        results['mape'] = mape
        results['r2'] = r2
        results['bias_mean'] = mean_bias
        results['bias_std'] = std_bias

        # 单任务预警分析
        if evaluation_system:
            warning_results = evaluation_system.single_task_warning_analysis(
                preds, trues, confidence_scores, cycle_numbers, task_name
            )
            results.update(warning_results)

    # ---------------- 总推理耗时输出 ----------------
    if total_infer_samples > 0:
        accelerator.print(f"[Val] 平均推理 {total_infer_samples} 样本 | {total_infer_time:.2f}s | {total_infer_time/total_infer_samples*1000:.2f} ms/sample")

    model.train()
    return results


class BatteryEvaluationAndWarning:
    """电池预测模型评估和预警系统"""

    def __init__(self, save_path: str = './evaluation_results'):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        # 在初始化时，阈值可以为空或默认值
        self.warning_thresholds = {
            'soc': {'critical': 0.02, 'warning': 0.01},
            'soh': {'critical': 0.02, 'warning': 0.01},
        }

        self.colors = {
            'normal': '#2E86AB',
            'warning': '#F18F01',
            'critical': '#C73E1D',
            'predicted': '#A23B72',
            'confidence': '#4CAF50'
        }

    def set_warning_thresholds(self, test_metrics: dict):
        """
        根据测试集指标动态设置预警阈值。
        """
        self.warning_thresholds = {
            'soc': {
                'critical': test_metrics['soc_mae'] * 2,
                'warning': test_metrics['soc_mae'] * 1.5
            },
            'soh': {
                'critical': test_metrics['soh_mae'] * 2,
                'warning': test_metrics['soh_mae'] * 1.5
            },
        }
        print(
            f"SOC 阈值: 严重={self.warning_thresholds['soc']['critical']:.4f}, 一般={self.warning_thresholds['soc']['warning']:.4f}")
        print(
            f"SOH 阈值: 严重={self.warning_thresholds['soh']['critical']:.4f}, 一般={self.warning_thresholds['soh']['warning']:.4f}")

    def comprehensive_warning_analysis(self, predictions_dict, actuals_dict,
                                       confidence_scores, cycle_numbers, battery_types, task_type):
        """综合预警分析"""
        results = {}

        # 1. 阈值预警检测
        threshold_warnings = self.threshold_warning_detection(predictions_dict)
        results['threshold_warnings'] = threshold_warnings

        # 2. 趋势恶化检测
        trend_warnings = self.trend_degradation_detection(predictions_dict, cycle_numbers)
        results['trend_warnings'] = trend_warnings

        # 3. 置信度分析
        confidence_analysis = self.confidence_analysis(confidence_scores, predictions_dict)
        results['confidence_analysis'] = confidence_analysis

        # 4. 异常检测
        anomaly_detection = self.anomaly_detection(predictions_dict, actuals_dict)
        results['anomaly_detection'] = anomaly_detection

        # 5. 生成可视化
        self.create_warning_visualizations(predictions_dict, actuals_dict,
                                           confidence_scores, cycle_numbers, task_type)

        return results

    def single_task_warning_analysis(self, predictions, actuals, confidence_scores,
                                     cycle_numbers, task_name):
        """单任务预警分析"""
        pred_dict = {task_name: predictions}
        actual_dict = {task_name: actuals}

        # 创建空的 battery_types 列表（单任务时不需要）
        battery_types = [0] * len(predictions)

        return self.comprehensive_warning_analysis(pred_dict, actual_dict,
                                                   confidence_scores, cycle_numbers,
                                                   battery_types, 'single')
    def threshold_warning_detection(self, predictions_dict):
        """阈值预警检测"""
        warnings = {}

        for task, preds in predictions_dict.items():
            if task in self.warning_thresholds:
                thresholds = self.warning_thresholds[task]

                critical_mask = preds < thresholds['critical']
                warning_mask = (preds < thresholds['warning']) & (preds >= thresholds['critical'])

                warnings[task] = {
                    'critical_count': np.sum(critical_mask),
                    'warning_count': np.sum(warning_mask),
                    'critical_percentage': np.sum(critical_mask) / len(preds) * 100,
                    'warning_percentage': np.sum(warning_mask) / len(preds) * 100,
                    'critical_indices': np.where(critical_mask)[0].tolist(),
                    'warning_indices': np.where(warning_mask)[0].tolist()
                }

        return warnings

    def trend_degradation_detection(self, predictions_dict, cycle_numbers):
        """趋势恶化检测"""
        trend_analysis = {}

        for task, preds in predictions_dict.items():
            # 计算滑动平均趋势
            window_size = min(10, len(preds) // 5)
            if window_size > 2:
                moving_avg = np.convolve(preds, np.ones(window_size) / window_size, mode='valid')

                # 计算趋势斜率
                if len(moving_avg) > 2:
                    x = np.arange(len(moving_avg))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, moving_avg)

                    # 判断恶化趋势
                    is_degrading = False
                    if task in ['soc', 'soh'] and slope < -0.5:  # SOC/SOH下降
                        is_degrading = True

                    trend_analysis[task] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'is_degrading': is_degrading,
                        'degradation_rate': abs(slope) if is_degrading else 0
                    }

        return trend_analysis

    def confidence_analysis(self, confidence_scores, predictions_dict):
        """置信度分析"""
        confidence_array = np.array(confidence_scores)

        analysis = {
            'mean_confidence': np.mean(confidence_array),
            'std_confidence': np.std(confidence_array),
            'low_confidence_threshold': 0.3,
            'low_confidence_count': np.sum(confidence_array < 0.3),
            'low_confidence_percentage': np.sum(confidence_array < 0.3) / len(confidence_array) * 100
        }

        return analysis

    def anomaly_detection(self, predictions_dict, actuals_dict):
        """异常检测"""
        anomalies = {}

        for task in predictions_dict.keys():
            if task in actuals_dict:
                preds = predictions_dict[task]
                actuals = actuals_dict[task]

                # 计算残差
                residuals = preds - actuals

                # 使用3-sigma规则检测异常
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                threshold = 3 * std_residual

                anomaly_mask = np.abs(residuals - mean_residual) > threshold

                anomalies[task] = {
                    'anomaly_count': np.sum(anomaly_mask),
                    'anomaly_percentage': np.sum(anomaly_mask) / len(residuals) * 100,
                    'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                    'max_positive_error': np.max(residuals),
                    'max_negative_error': np.min(residuals),
                    'mean_absolute_residual': np.mean(np.abs(residuals))
                }

        return anomalies

    def create_warning_visualizations(self, predictions_dict, actuals_dict,
                                      confidence_scores, cycle_numbers, task_type):
        """创建预警可视化图表"""
        try:
            # 1. 综合预警仪表板
            self.plot_warning_dashboard(predictions_dict, actuals_dict,
                                        confidence_scores, cycle_numbers)

            # 2. 置信度分析图
            self.plot_confidence_analysis(confidence_scores, cycle_numbers)

            # 3. 趋势分析图
            self.plot_trend_analysis(predictions_dict, cycle_numbers)

            # 4. 阈值预警图
            self.plot_threshold_warnings(predictions_dict, cycle_numbers)
        except Exception as e:
            print(f"可视化生成出错，跳过可视化步骤: {e}")

    def plot_warning_dashboard(self, predictions_dict, actuals_dict,
                               confidence_scores, cycle_numbers):
        """绘制综合预警仪表板"""
        n_tasks = len(predictions_dict)
        fig, axes = plt.subplots(2, n_tasks, figsize=(6 * n_tasks, 10))
        if n_tasks == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Battery Early Warning System Dashboard', fontsize=16, fontweight='bold')#电池预警系统仪表板

        for i, (task, preds) in enumerate(predictions_dict.items()):
            actuals = actuals_dict.get(task, preds)

            # 上排：预测值与阈值
            ax_upper = axes[0, i]
            # 修复：确保 cycles 是 numpy 数组
            cycles = np.array(cycle_numbers[:len(preds)])

            ax_upper.plot(cycles, preds, color=self.colors['predicted'],
                          linewidth=2, label='Predicted value')
            ax_upper.plot(cycles, actuals, color=self.colors['normal'],
                          linewidth=2, alpha=0.7, label='True value')

            # 添加预警阈值线
            if task in self.warning_thresholds:
                thresholds = self.warning_thresholds[task]
                ax_upper.axhline(y=thresholds['warning'], color=self.colors['warning'],
                                 linestyle='--', alpha=0.8, label='Warning threshold')#警告阈值
                ax_upper.axhline(y=thresholds['critical'], color=self.colors['critical'],
                                 linestyle='--', alpha=0.8, label='Severe Threshold')#严重阈值

                # 标记预警区域
                ax_upper.fill_between(cycles, 0, thresholds['critical'],
                                      color=self.colors['critical'], alpha=0.1)
                ax_upper.fill_between(cycles, thresholds['critical'], thresholds['warning'],
                                      color=self.colors['warning'], alpha=0.1)

            ax_upper.set_title(f'{task.upper()} Early warning monitoring')#预警监控
            ax_upper.set_xlabel('Number of cycles')#循环次数
            ax_upper.set_ylabel(f'{task.upper()} value')
            ax_upper.legend()
            ax_upper.grid(True, alpha=0.3)

            # 下排：置信度分析
            ax_lower = axes[1, i]
            conf_subset = confidence_scores[:len(preds)]

            # 置信度散点图
            scatter = ax_lower.scatter(cycles, preds, c=conf_subset,
                                       cmap='RdYlGn', s=30, alpha=0.7)

            # 添加置信度颜色条
            cbar = plt.colorbar(scatter, ax=ax_lower)
            cbar.set_label('Confidence')#置信度

            ax_lower.set_title(f'{task.upper()} Confidence Analysis')#置信度分析
            ax_lower.set_xlabel('Number of cycles')
            ax_lower.set_ylabel(f'{task.upper()} value')
            ax_lower.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'warning_dashboard.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

    def plot_confidence_analysis(self, confidence_scores, cycle_numbers):
        """绘制置信度分析图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        confidence_array = np.array(confidence_scores)
        cycles = np.array(cycle_numbers[:len(confidence_array)])  # 修复：确保是numpy数组

        # 1. 置信度时序图
        axes[0].plot(cycles, confidence_array, color=self.colors['confidence'], linewidth=2)
        axes[0].axhline(y=0.3, color=self.colors['critical'], linestyle='--',
                        alpha=0.8, label='Low confidence threshold')#低置信度阈值
        axes[0].fill_between(cycles, 0, 0.3, color=self.colors['critical'], alpha=0.1)
        axes[0].set_title('Temporal changes in confidence')#置信度时序变化
        axes[0].set_xlabel('Number of cycles')
        axes[0].set_ylabel('Confidence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 置信度分布直方图
        axes[1].hist(confidence_array, bins=30, color=self.colors['confidence'],
                     alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(confidence_array), color='red', linestyle='--',
                        linewidth=2, label=f'mean: {np.mean(confidence_array):.3f}')
        axes[1].axvline(0.3, color=self.colors['critical'], linestyle='--',
                        linewidth=2, label='Low confidence threshold')#低置信度阈值
        axes[1].set_title('Confidence distribution')#置信度分布
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 置信度统计信息
        axes[2].axis('off')
        #置信度统计信息
        stats_text = f"""Confidence statistics:

        平均置信度: {np.mean(confidence_array):.3f}
        置信度标准差: {np.std(confidence_array):.3f}
        最低置信度: {np.min(confidence_array):.3f}
        最高置信度: {np.max(confidence_array):.3f}

        低置信度样本数: {np.sum(confidence_array < 0.3)}
        低置信度比例: {np.sum(confidence_array < 0.3) / len(confidence_array) * 100:.1f}%

        预警建议:
        {'需要关注低置信度预测结果' if np.sum(confidence_array < 0.3) / len(confidence_array) > 0.1 else '置信度整体良好'}
        """

        axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'confidence_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

    def plot_trend_analysis(self, predictions_dict, cycle_numbers):
        """绘制趋势分析图"""
        n_tasks = len(predictions_dict)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
        if n_tasks == 1:
            axes = [axes]

        fig.suptitle('Battery performance trend analysis', fontsize=16, fontweight='bold')#电池性能趋势分析

        for i, (task, preds) in enumerate(predictions_dict.items()):
            cycles = np.array(cycle_numbers[:len(preds)])  # 修复：确保是numpy数组

            # 原始数据
            axes[i].scatter(cycles, preds, alpha=0.5, s=20, color=self.colors['normal'])

            # 拟合趋势线
            if len(preds) > 2:
                z = np.polyfit(cycles, preds, 1)
                p = np.poly1d(z)
                axes[i].plot(cycles, p(cycles), color=self.colors['critical'],
                             linewidth=3, alpha=0.8, label=f'Trend lines (Slope: {z[0]:.4f})')#斜率

                # 计算滑动平均
                window_size = min(10, len(preds) // 5)
                if window_size > 2:
                    moving_avg = np.convolve(preds, np.ones(window_size) / window_size, mode='valid')
                    moving_cycles = cycles[:len(moving_avg)]
                    axes[i].plot(moving_cycles, moving_avg, color=self.colors['warning'],
                                 linewidth=2, alpha=0.8, label=f'Moving average (window: {window_size})')#滑动平均

            axes[i].set_title(f'{task.upper()} Performance Trends')#性能趋势
            axes[i].set_xlabel('Number of cycles')
            axes[i].set_ylabel(f'{task.upper()} value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'trend_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

    def plot_threshold_warnings(self, predictions_dict, cycle_numbers):
        """绘制阈值预警图"""
        n_tasks = len(predictions_dict)
        fig, axes = plt.subplots(2, n_tasks, figsize=(6 * n_tasks, 8))
        if n_tasks == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Threshold warning analysis', fontsize=16, fontweight='bold')#阈值预警分析

        for i, (task, preds) in enumerate(predictions_dict.items()):
            # 修复：确保 cycles 和 preds 都是 numpy 数组
            cycles = np.array(cycle_numbers[:len(preds)])
            preds = np.array(preds)

            # 上排：阈值状态图
            ax_upper = axes[0, i]
            ax_upper.plot(cycles, preds, color=self.colors['normal'], linewidth=2)

            if task in self.warning_thresholds:
                thresholds = self.warning_thresholds[task]

                # 预警区域标记
                critical_mask = preds < thresholds['critical']
                warning_mask = (preds < thresholds['warning']) & (preds >= thresholds['critical'])
                normal_mask = preds >= thresholds['warning']

                # 修复：使用布尔索引前确保数组类型匹配
                if np.any(critical_mask):
                    ax_upper.scatter(cycles[critical_mask], preds[critical_mask],
                                     color=self.colors['critical'], s=50, alpha=0.8, label='serious')
                if np.any(warning_mask):
                    ax_upper.scatter(cycles[warning_mask], preds[warning_mask],
                                     color=self.colors['warning'], s=50, alpha=0.8, label='warning')
                if np.any(normal_mask):
                    ax_upper.scatter(cycles[normal_mask], preds[normal_mask],
                                     color=self.colors['normal'], s=50, alpha=0.8, label='normal')

                # 阈值线
                ax_upper.axhline(y=thresholds['warning'], color=self.colors['warning'],
                                 linestyle='--', alpha=0.8)
                ax_upper.axhline(y=thresholds['critical'], color=self.colors['critical'],
                                 linestyle='--', alpha=0.8)

            ax_upper.set_title(f'{task.upper()} Threshold warning status')#阈值预警状态
            ax_upper.set_xlabel('Number of cycles')
            ax_upper.set_ylabel(f'{task.upper()} value')
            ax_upper.legend()
            ax_upper.grid(True, alpha=0.3)

            # 下排：预警统计
            ax_lower = axes[1, i]
            if task in self.warning_thresholds:
                thresholds = self.warning_thresholds[task]
                critical_count = np.sum(preds < thresholds['critical'])
                warning_count = np.sum((preds < thresholds['warning']) &
                                       (preds >= thresholds['critical']))
                normal_count = np.sum(preds >= thresholds['warning'])

                categories = ['normal', 'warning', 'serious']
                counts = [normal_count, warning_count, critical_count]
                colors = [self.colors['normal'], self.colors['warning'], self.colors['critical']]

                bars = ax_lower.bar(categories, counts, color=colors, alpha=0.8)

                # 添加百分比标签
                total = len(preds)
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    percentage = count / total * 100
                    ax_lower.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                                  f'{count}\n({percentage:.1f}%)',
                                  ha='center', va='bottom', fontweight='bold')

            ax_lower.set_title(f'{task.upper()} Early warning distribution statistics')#预警分布统计
            ax_lower.set_ylabel('Sample size')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'threshold_warnings.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

    def generate_warning_report(self, results):
        """生成预警报告"""
        report = []
        report.append("=" * 60)
        report.append("Battery prediction system early warning report")#电池预测系统预警报告
        report.append("=" * 60)

        # 阈值预警总结
        if 'threshold_warnings' in results:
            report.append("\n1. Threshold warning analysis:")#阈值预警分析
            for task, warnings in results['threshold_warnings'].items():
                report.append(f"\n{task.upper()}:")
                report.append(f"  - Serious warning: {warnings['critical_count']} times"
                              f"({warnings['critical_percentage']:.1f}%)")
                report.append(f"  - General warning: {warnings['warning_count']} times"
                              f"({warnings['warning_percentage']:.1f}%)")

        # 趋势分析总结
        if 'trend_warnings' in results:
            report.append("\n2. Performance trend analysis:")#性能趋势分析
            for task, trend in results['trend_warnings'].items():
                if trend['is_degrading']:
                    report.append(f"\n{task.upper()}: ⚠️ Performance deterioration trend detected")#检测到性能恶化趋势
                    report.append(f"  - Deterioration rate: {trend['degradation_rate']:.4f} Units/cycle")#单位/循环
                    report.append(f"  - Trend credibility: {trend['r_squared']:.3f}")
                else:
                    report.append(f"\n{task.upper()}: ✅Performance trend is normal")#性能趋势正常

        # 置信度分析总结
        if 'confidence_analysis' in results:
            conf = results['confidence_analysis']
            report.append(f"\n3. Forecast confidence analysis:")#预测置信度分析
            report.append(f"  - Average confidence: {conf['mean_confidence']:.3f}")#平均置信度
            report.append(f"  - Low confidence samples: {conf['low_confidence_count']}"#低置信度样本
                          f"({conf['low_confidence_percentage']:.1f}%)")

        # 异常检测总结
        if 'anomaly_detection' in results:
            report.append("\n4. Anomaly detection results:")#异常检测结果
            for task, anomaly in results['anomaly_detection'].items():
                report.append(f"\n{task.upper()}:")
                report.append(f"  - Number of abnormal samples: {anomaly['anomaly_count']} "#异常样本数
                              f"({anomaly['anomaly_percentage']:.1f}%)")
                report.append(f"  - Mean absolute error: {anomaly['mean_absolute_residual']:.4f}")

        report.append("\n" + "=" * 60)

        # 保存报告
        with open(os.path.join(self.save_path, 'warning_report.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        return '\n'.join(report)


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