#!/usr/bin/env python3
"""
实验结果分析脚本

这个脚本可以：
1. 加载和解析实验结果
2. 比较不同超参数的性能
3. 生成可视化图表
4. 找出最佳配置
5. 导出实验报告

使用方法:
python analyze_experiments.py
python analyze_experiments.py --top 5
python analyze_experiments.py --filter optimizer=ADAMW
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentAnalyzer:
    def __init__(self, results_file: str = None):
        self.results_dir = Path("../experiment")
        self.results_file = results_file or self.results_dir / "experiment_results.csv"

        # 确保结果目录存在
        self.results_dir.mkdir(exist_ok=True)

        # 加载结果
        self.df = self.load_results()

    def load_results(self) -> pd.DataFrame:
        """加载实验结果"""
        if not self.results_file.exists():
            print(f"❌ Results file not found: {self.results_file}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.results_file)
            print(f"✅ Loaded {len(df)} experiment results")

            # 清理数据
            df = self.clean_data(df)
            return df

        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 转换数值列
        numeric_columns = ['duration_hours', 'config_lr', 'config_symunet_pretrain_width']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 过滤成功的实验
        df = df[df['status'] == 'SUCCESS'].copy()

        # 添加超参数列的简化名称
        param_mappings = {
            'config_optimizer': 'optimizer',
            'config_scheduler': 'scheduler',
            'config_lr': 'learning_rate',
            'config_symunet_pretrain_width': 'model_width',
            'config_loss': 'loss_function'
        }

        for old_col, new_col in param_mappings.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        return df

    def show_summary(self):
        """显示实验摘要"""
        if self.df.empty:
            print("❌ No data to analyze")
            return

        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        total_experiments = len(self.df)
        avg_time = self.df['duration_hours'].mean()
        total_time = self.df['duration_hours'].sum()

        print(f"📈 Total successful experiments: {total_experiments}")
        print(f"⏱️ Average training time: {avg_time:.2f} hours")
        print(f"🕐 Total training time: {total_time:.2f} hours")
        print(f"📅 First experiment: {self.df['start_time'].min()}")
        print(f"📅 Last experiment: {self.df['start_time'].max()}")

    def show_top_experiments(self, top_n: int = 10):
        """显示最佳实验"""
        if self.df.empty:
            print("❌ No data to analyze")
            return

        print(f"\n🏆 TOP {top_n} EXPERIMENTS (by ID)")
        print("-" * 100)

        # 选择要显示的列
        display_columns = [
            'experiment_id', 'experiment_name', 'duration_hours',
            'optimizer', 'scheduler', 'learning_rate', 'model_width', 'loss_function'
        ]

        available_columns = [col for col in display_columns if col in self.df.columns]
        top_experiments = self.df.head(top_n)[available_columns]

        # 格式化显示
        for col in ['duration_hours', 'learning_rate']:
            if col in top_experiments.columns:
                if col == 'learning_rate':
                    top_experiments[col] = top_experiments[col].apply(lambda x: f"{x:.1e}" if pd.notna(x) else "N/A")
                else:
                    top_experiments[col] = top_experiments[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        print(top_experiments.to_string(index=False))

    def analyze_hyperparameters(self):
        """分析超参数影响"""
        if self.df.empty:
            print("❌ No data to analyze")
            return

        print(f"\n🔍 HYPERPARAMETER ANALYSIS")
        print(f"{'='*80}")

        # 分析各个超参数
        params_to_analyze = ['optimizer', 'scheduler', 'learning_rate', 'model_width', 'loss_function']

        for param in params_to_analyze:
            if param not in self.df.columns:
                continue

            print(f"\n📊 {param.upper().replace('_', ' ')} Analysis:")
            print("-" * 50)

            param_stats = self.df.groupby(param).agg({
                'duration_hours': ['count', 'mean', 'std'],
                'experiment_id': 'count'
            }).round(2)

            param_stats.columns = ['count', 'avg_duration', 'std_duration', 'total_experiments']
            print(param_stats)

    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """创建可视化图表"""
        if self.df.empty:
            print("❌ No data to visualize")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n📈 Creating visualizations...")

        # 1. 训练时间分布
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['duration_hours'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Training Time Distribution')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'training_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 超参数vs训练时间
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 优化器vs训练时间
        if 'optimizer' in self.df.columns:
            sns.boxplot(data=self.df, x='optimizer', y='duration_hours', ax=axes[0,0])
            axes[0,0].set_title('Optimizer vs Training Time')
            axes[0,0].tick_params(axis='x', rotation=45)

        # 调度器vs训练时间
        if 'scheduler' in self.df.columns:
            sns.boxplot(data=self.df, x='scheduler', y='duration_hours', ax=axes[0,1])
            axes[0,1].set_title('Scheduler vs Training Time')
            axes[0,1].tick_params(axis='x', rotation=45)

        # 学习率vs训练时间
        if 'learning_rate' in self.df.columns:
            sns.scatterplot(data=self.df, x='learning_rate', y='duration_hours', ax=axes[1,0])
            axes[1,0].set_title('Learning Rate vs Training Time')
            axes[1,0].set_xscale('log')

        # 模型宽度vs训练时间
        if 'model_width' in self.df.columns:
            sns.scatterplot(data=self.df, x='model_width', y='duration_hours', ax=axes[1,1])
            axes[1,1].set_title('Model Width vs Training Time')

        plt.tight_layout()
        plt.savefig(output_path / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 相关性热图
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Hyperparameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Visualizations saved to {output_path}")

    def export_report(self, output_file: str = "experiment_report.html"):
        """导出实验报告"""
        if self.df.empty:
            print("❌ No data to export")
            return

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 SymUNet Experiment Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <ul>
                    <li><strong>Total Experiments:</strong> {len(self.df)}</li>
                    <li><strong>Average Training Time:</strong> {self.df['duration_hours'].mean():.2f} hours</li>
                    <li><strong>Total Training Time:</strong> {self.df['duration_hours'].sum():.2f} hours</li>
                    <li><strong>Fastest Experiment:</strong> {self.df['duration_hours'].min():.2f} hours</li>
                    <li><strong>Slowest Experiment:</strong> {self.df['duration_hours'].max():.2f} hours</li>
                </ul>
            </div>

            <div class="section">
                <h2>🏆 Top Experiments</h2>
                {self.df.head(10)[['experiment_name', 'optimizer', 'scheduler', 'learning_rate', 'duration_hours']].to_html(index=False)}
            </div>

            <div class="section">
                <h2>📈 Parameter Analysis</h2>
                <h3>Optimizer Performance</h3>
                {self.df.groupby('optimizer')['duration_hours'].agg(['count', 'mean']).to_html()}

                <h3>Scheduler Performance</h3>
                {self.df.groupby('scheduler')['duration_hours'].agg(['count', 'mean']).to_html()}
            </div>
        </body>
        </html>
        """

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Report exported to {output_file}")

    def filter_experiments(self, filter_expr: str):
        """过滤实验"""
        if self.df.empty:
            print("❌ No data to filter")
            return pd.DataFrame()

        try:
            # 简单的过滤语法：param=value
            param, value = filter_expr.split('=')
            filtered_df = self.df[self.df[param].astype(str) == value]
            print(f"✅ Filtered to {len(filtered_df)} experiments with {param}={value}")
            return filtered_df

        except Exception as e:
            print(f"❌ Error filtering: {e}")
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="实验结果分析工具")
    parser.add_argument("--results-file", type=str, help="结果文件路径")
    parser.add_argument("--top", type=int, default=10, help="显示最佳实验数量")
    parser.add_argument("--filter", type=str, help="过滤条件，格式：param=value")
    parser.add_argument("--visualize", action="store_true", help="生成可视化图表")
    parser.add_argument("--export", type=str, help="导出报告文件名")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="输出目录")

    args = parser.parse_args()

    # 初始化分析器
    analyzer = ExperimentAnalyzer(args.results_file)

    if analyzer.df.empty:
        print("❌ No experiment results found")
        return

    # 显示摘要
    analyzer.show_summary()

    # 过滤实验
    if args.filter:
        filtered_df = analyzer.filter_experiments(args.filter)
        if not filtered_df.empty:
            analyzer.df = filtered_df
            analyzer.show_summary()

    # 显示最佳实验
    analyzer.show_top_experiments(args.top)

    # 分析超参数
    analyzer.analyze_hyperparameters()

    # 生成可视化
    if args.visualize:
        analyzer.create_visualizations(args.output_dir)

    # 导出报告
    if args.export:
        analyzer.export_report(args.export)

    print(f"\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
