#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理验证脚本
检查所有数据集的预处理状态并生成验证报告
"""

import os
import pandas as pd
from pathlib import Path

# 定义项目根目录
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# 定义数据集列表
DATASETS = [
    'ipk_out_raw',
    'pepper',
    'pepper_10611831',
    'pepper_11955216',
    'pepper_7268809'
]

# 定义期望的处理后文件列表
EXPECTED_OUTPUT_FILES = [
    'X.csv',
    'y.csv',
    'pca_covariates.csv',
    'variant_manifest.csv',
    'sample_map.csv',
    'qc_report.txt'
]

def check_preprocessing_scripts(dataset):
    """检查数据集的预处理脚本是否存在"""
    scripts_dir = PROJECT_ROOT / '01_preprocessing' / dataset
    scripts = list(scripts_dir.glob('*.py'))
    return [script.name for script in scripts]

def check_processed_data_files(dataset):
    """检查数据集的处理后文件是否存在"""
    data_dir = PROJECT_ROOT / '02_processed_data' / dataset
    existing_files = []
    missing_files = []
    
    # 检查目录是否存在
    if not data_dir.exists():
        return [], EXPECTED_OUTPUT_FILES
    
    # 检查每个期望的文件
    for file in EXPECTED_OUTPUT_FILES:
        file_path = data_dir / file
        if file_path.exists() and file_path.stat().st_size > 0:
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

def check_log_files(dataset):
    """检查日志文件是否存在"""
    logs_dir = PROJECT_ROOT / 'logs'
    log_patterns = [
        f"{dataset}_prep.log",
        f"{dataset}_complete_prep.log"
    ]
    
    found_logs = []
    for pattern in log_patterns:
        log_file = logs_dir / pattern
        if log_file.exists():
            found_logs.append(log_file.name)
    
    return found_logs

def determine_preprocessing_status(existing_files):
    """根据存在的文件确定预处理状态"""
    if len(existing_files) == len(EXPECTED_OUTPUT_FILES):
        return "✅ Complete"
    elif len(existing_files) > 0:
        return "⚠️ Partial"
    else:
        return "❌ Incomplete"

def generate_verification_report():
    """生成预处理验证报告"""
    report_data = []
    
    for dataset in DATASETS:
        # 检查预处理脚本
        preprocessing_scripts = check_preprocessing_scripts(dataset)
        
        # 检查处理后的数据文件
        existing_files, missing_files = check_processed_data_files(dataset)
        
        # 检查日志文件
        log_files = check_log_files(dataset)
        
        # 确定预处理状态
        status = determine_preprocessing_status(existing_files)
        
        # 记录问题和备注
        issues = ", ".join(missing_files) if missing_files else "None"
        notes = f"预处理脚本: {', '.join(preprocessing_scripts)}\n日志文件: {', '.join(log_files)}\n存在的处理文件: {', '.join(existing_files)}"
        
        # 添加到报告数据
        report_data.append({
            'Dataset Name': dataset,
            'Preprocessing Status': status,
            'Missing or problematic files': issues,
            'Notes or errors found': notes
        })
    
    # 创建DataFrame
    df_report = pd.DataFrame(report_data)
    
    # 生成Markdown报告
    markdown_report = """# 预处理验证报告

本报告验证了项目中所有数据集的预处理状态。

## 验证结果摘要

| Dataset Name | Preprocessing Status | Missing or problematic files | Notes or errors found |
|-------------|----------------------|------------------------------|----------------------|
"""
    
    # 添加表格行
    for _, row in df_report.iterrows():
        # 先处理notes中的换行符
        notes_formatted = row['Notes or errors found'].replace('\n', '<br>')
        markdown_report += f"| {row['Dataset Name']} | {row['Preprocessing Status']} | {row['Missing or problematic files']} | {notes_formatted} |\n"
    
    markdown_report += """\n## 详细分析

"""
    
    # 添加详细分析
    for _, row in df_report.iterrows():
        markdown_report += f"""### {row['Dataset Name']}

**预处理状态**: {row['Preprocessing Status']}

**缺失文件**: {row['Missing or problematic files']}

**详细信息**:
```
{row['Notes or errors found']}
```

"""
    
    markdown_report += """\n## 结论

"""
    
    # 添加结论
    complete_count = len(df_report[df_report['Preprocessing Status'] == "✅ Complete"])
    partial_count = len(df_report[df_report['Preprocessing Status'] == "⚠️ Partial"])
    incomplete_count = len(df_report[df_report['Preprocessing Status'] == "❌ Incomplete"])
    
    markdown_report += f"""在 {len(DATASETS)} 个数据集中：
- ✅ {complete_count} 个数据集已完全预处理
- ⚠️ {partial_count} 个数据集部分预处理
- ❌ {incomplete_count} 个数据集未预处理

"""
    
    if partial_count > 0 or incomplete_count > 0:
        markdown_report += """## 建议的后续步骤

"""
        
        for _, row in df_report.iterrows():
            if row['Preprocessing Status'] in ["⚠️ Partial", "❌ Incomplete"]:
                scripts_dir = PROJECT_ROOT / '01_preprocessing' / row['Dataset Name']
                scripts = check_preprocessing_scripts(row['Dataset Name'])
                
                if scripts:
                    markdown_report += f"""### {row['Dataset Name']}

建议重新运行以下预处理脚本：
```
cd {scripts_dir}
python {scripts[0]}
```

"""
                else:
                    markdown_report += f"""### {row['Dataset Name']}

警告：未找到预处理脚本，请检查该数据集是否需要特殊处理。

"""
    
    # 保存报告
    report_path = PROJECT_ROOT / "preprocessing_verification_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"验证报告已保存到: {report_path}")
    return df_report

if __name__ == "__main__":
    df = generate_verification_report()
    print(df)