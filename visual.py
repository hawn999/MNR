import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def visualize_error_trends(csv_path="/home/scxhc1/prediction_errors.csv",
                           layers_to_plot=None,
                           output_filename="error_comparison_smoothed.png",
                           smoothing_window=100):  # <-- 新增：平滑窗口大小
    """
    读取并可视化prediction_errors.csv文件, 对比正确与错误答案的误差，并对曲线进行平滑处理。

    Args:
        csv_path (str): prediction_errors.csv文件的路径。
        layers_to_plot (list, optional): 要显示的层列表。如果为 None, 则显示所有层。
        output_filename (str): 输出图片的文件名。
        smoothing_window (int, optional): 滑动平均的窗口大小。设为 None 或 <= 1 则不进行平滑。
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 '{csv_path}', 共 {len(df)} 条记录。")

        if layers_to_plot is not None:
            print(f"筛选数据，只显示 Layers: {layers_to_plot}")
            df = df[df['layer'].isin(layers_to_plot)]

        if df.empty:
            print("筛选后数据为空，无法生成图表。")
            return

        df['comparison'] = df['phase'] + ' (' + df['status'] + ')'
        df['layer'] = df['layer'].astype('category')
        df['comparison'] = df['comparison'].astype('category')

        # --- 主要改动 1: 计算滑动平均值 ---
        y_column = 'mean_abs_error'
        if smoothing_window and smoothing_window > 1:
            print(f"对误差数据进行平滑处理，窗口大小: {smoothing_window}")
            # 使用 groupby().transform() 来确保在每个独立的曲线内部进行滑动平均
            df['smoothed_error'] = df.groupby(['comparison', 'layer'])['mean_abs_error'] \
                .transform(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean())
            y_column = 'smoothed_error'  # 后面绘图时使用平滑后的数据列

        plt.figure(figsize=(16, 9))

        # --- 主要改动 2: 使用平滑后的数据列进行绘图 ---
        sns.lineplot(
            data=df,
            x='step',
            y=y_column,  # 使用 y_column 变量
            hue='comparison',
            style='layer',
            alpha=0.9,
            linewidth=2  # 将线条加粗一点，看得更清楚
        )

        # (可选) 如果想同时显示原始的抖动数据作为背景，可以取消下面的注释
        # sns.lineplot(
        #     data=df, x='step', y='mean_abs_error', hue='comparison', style='layer',
        #     alpha=0.2, legend=False
        # )

        # --- 图表美化 ---
        title = f'Smoothed Correct vs Incorrect Prediction Error Trend'
        if layers_to_plot is not None:
            title += f' for Layers {layers_to_plot}'
        plt.title(title, fontsize=20, pad=20)
        plt.xlabel('Training Step', fontsize=14)
        plt.ylabel('Mean Absolute Error (Smoothed)', fontsize=14)
        plt.yscale('log')
        plt.legend(title='Phase (Status) / Layer', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x / 1000)}k' if x > 0 else '0'))
        plt.tight_layout()

        plt.savefig(output_filename, dpi=300)
        print(f"\n平滑后的可视化图表已保存为 '{output_filename}'")

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{csv_path}'。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")



path = "/home/scxhc1/MNR_IJCAI25/ckpts/mlp_raven_error_16_5_2025-09-25-23-25-43_RAVEN-predrnet_original_raven-prb5-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/prediction_errors.csv"
visualize_error_trends(path, layers_to_plot=[4], output_filename='mlp_r_2_smooth100_16_5.png', smoothing_window=100)
visualize_error_trends(path, layers_to_plot=[0, 1, 2, 3, 4], output_filename='mlp_r_16_5.png', smoothing_window=100)

path = "/home/scxhc1/MNR_IJCAI25/ckpts/mlp_raven_error_32_2025-09-25-23-23-28_RAVEN-predrnet_original_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/prediction_errors.csv"
visualize_error_trends(path, layers_to_plot=[2], output_filename='mlp_r_2_smooth100_32.png', smoothing_window=100)
visualize_error_trends(path, layers_to_plot=[0, 1, 2], output_filename='mlp_r_32.png', smoothing_window=100)

path = "/home/scxhc1/MNR_IJCAI25/ckpts/mlp_raven_error_24_2025-09-25-23-23-02_RAVEN-predrnet_original_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/prediction_errors.csv"
visualize_error_trends(path, layers_to_plot=[2], output_filename='mlp_r_2_smooth100_24.png', smoothing_window=100)
visualize_error_trends(path, layers_to_plot=[0, 1, 2], output_filename='mlp_r_24.png', smoothing_window=100)
# path = "/home/scxhc1/MNR_IJCAI25/ckpts/v3_raven_error_01_RAVEN-predrnet_original_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/prediction_errors.csv"
# visualize_error_trends(path, layers_to_plot=[2], output_filename='r_2_smooth200.png', smoothing_window=200)
# # visualize_error_trends(path, layers_to_plot=[0, 1, 2], output_filename='r.png')
# #
# path = "/home/scxhc1/MNR_IJCAI25/ckpts/v3_ravenf_error_01_RAVEN-FAIR-predrnet_original_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/prediction_errors.csv"
# visualize_error_trends(path, layers_to_plot=[2], output_filename='rf_2_smooth200.png', smoothing_window=200)
# # visualize_error_trends(path, layers_to_plot=[0, 1, 2], output_filename='rf.png')