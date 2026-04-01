import os
import json
import logging
import time
import numpy as np
import pandas as pd
import joblib
import sys
from datetime import datetime
import argparse

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Unified modeling pipeline for SNP datasets')
parser.add_argument('--dataset', choices=['pepper', 'pepper_10611831', 'ipk_out_raw'], required=True,
                    help='Dataset to process')
args = parser.parse_args()

def setup_config():
    """设置配置信息"""
    # 创建唯一的run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 基础配置
    base_config = {
        "DATASET": args.dataset,
        "OVERWRITE_PREVIOUS": False,
        "REQUIRE_XGBOOST": False,
        "REQUIRE_LIGHTGBM": False,
        "RANDOM_SEED": 42,
        "OUTDIR": f"03_modeling_results/{args.dataset}",
        "PROCESSED": f"02_processed_data/{args.dataset}",
        "OUTER_CV_FOLDS": 5,
        "INNER_CV_FOLDS": 3,
        "MAX_FEATURES": 10000,  # 将在代码中根据需要调整
        "TARGET_COLUMN": None,  # 将在运行时确定
        "RUN_ID": run_id,
        "RUN_TYPE": "rerun"  # 标记为重新运行
    }
    
    # 根据数据集设置特定的目标列
    if args.dataset == 'ipk_out_raw':
        # ipk_out_raw需要使用真实的数值型表型
        base_config["TARGET_COLUMN"] = "YR_LS"  # 默认目标列
    elif args.dataset == 'pepper':
        base_config["TARGET_COLUMN"] = "Yield_BV"
    elif args.dataset == 'pepper_10611831':
        base_config["TARGET_COLUMN"] = "Yield"
    
    return base_config

# 设置配置
CONFIG = setup_config()

# 设置随机种子
np.random.seed(CONFIG["RANDOM_SEED"])

# 创建目录（如果不存在）
for subdir in ["logs", "models", "metrics", "plots"]:
    os.makedirs(os.path.join(CONFIG["OUTDIR"], subdir), exist_ok=True)

# 设置日志
log_file = os.path.join(CONFIG["OUTDIR"], "logs", f"pipeline_{CONFIG['RUN_ID']}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 导入sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR

# 尝试导入XGBoost
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    logging.info("XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    if CONFIG["REQUIRE_XGBOOST"]:
        logging.error("XGBoost required but not available")
        sys.exit(1)
    else:
        logging.warning("XGBoost not available, will skip")

# 尝试导入LightGBM
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    logging.info("LightGBM is available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    if CONFIG["REQUIRE_LIGHTGBM"]:
        logging.error("LightGBM required but not available")
        sys.exit(1)
    else:
        logging.warning("LightGBM not available, will skip")

# Import model registry
from model_registry import get_model_configs

def log_environment():
    """记录环境信息"""
    logging.info(f"Logging environment information for dataset: {CONFIG['DATASET']}")
    import platform
    import sklearn
    
    env_info = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "system": platform.system(),
        "timestamp": datetime.now().isoformat(),
        "pipeline_type": "unified",
        "run_id": CONFIG["RUN_ID"],
        "run_type": CONFIG["RUN_TYPE"]
    }
    
    if XGBOOST_AVAILABLE:
        import xgboost
        env_info["xgboost_version"] = xgboost.__version__
        
    if LIGHTGBM_AVAILABLE:
        import lightgbm
        env_info["lightgbm_version"] = lightgbm.__version__
    
    with open(os.path.join(CONFIG["OUTDIR"], "logs", f"env_{CONFIG['RUN_ID']}.txt"), "w") as f:
        for key, value in env_info.items():
            f.write(f"{key}: {value}\n")
    
    with open(os.path.join(CONFIG["OUTDIR"], "logs", f"config_{CONFIG['RUN_ID']}.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

def preflight_checks():
    """执行飞行前检查"""
    logging.info("Performing pre-flight checks")
    
    # 检查处理后的数据目录
    if not os.path.exists(CONFIG["PROCESSED"]):
        logging.error(f"Processed data directory not found: {CONFIG['PROCESSED']}")
        return False
    
    # 检查必需的文件
    required_files = ["X.csv", "y.csv", "pca_covariates.csv"]
    for file in required_files:
        file_path = os.path.join(CONFIG["PROCESSED"], file)
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            return False
    
    logging.info("Pre-flight checks passed")
    return True

def load_and_align_data():
    """加载并对齐数据，确保样本ID一致"""
    logging.info(f"Loading and aligning data for {CONFIG['DATASET']}")
    
    # 加载基因型数据 (X)
    X_path = os.path.join(CONFIG["PROCESSED"], "X.csv")
    logging.info(f"Loading X from {X_path}")
    X_df = pd.read_csv(X_path)
    
    # 处理第一行
    if 'POS' in X_df.columns or 'REF' in X_df.columns:
        X_df = X_df.iloc[1:].reset_index(drop=True)
    
    # 确保Sample_ID列存在
    if 'Sample_ID' not in X_df.columns:
        logging.error("Sample_ID column not found in X.csv")
        sys.exit(1)
    
    # 加载表型数据 (y)
    y_path = os.path.join(CONFIG["PROCESSED"], "y.csv")
    logging.info(f"Loading y from {y_path}")
    y_df = pd.read_csv(y_path)
    # 如果是ipk_out_raw，尝试通过sample_id_map.csv对齐原始表型到Sample_ID
    if CONFIG["DATASET"] == 'ipk_out_raw':
        map_path = os.path.join(CONFIG["PROCESSED"], "sample_id_map.csv")
        raw_pheno_path = os.path.join("ipk_out_raw", "Geno_IDs_and_Phenotypes.txt")
        if os.path.exists(map_path) and os.path.exists(raw_pheno_path):
            try:
                map_df = pd.read_csv(map_path)
                raw_pheno_df = pd.read_csv(raw_pheno_path, sep='\t')
                if {'GBS_BIOSAMPLE_ID', 'Sample_ID'}.issubset(map_df.columns):
                    y_df = raw_pheno_df.merge(
                        map_df[['GBS_BIOSAMPLE_ID', 'Sample_ID']],
                        on='GBS_BIOSAMPLE_ID', how='left'
                    )
                    y_df = y_df[y_df['Sample_ID'].notna()].reset_index(drop=True)
                    logging.info(f"Applied sample_id_map.csv; mapped {len(y_df)} rows with real target columns")
                else:
                    logging.warning("sample_id_map.csv missing required columns ['GBS_BIOSAMPLE_ID','Sample_ID']; using processed y.csv")
            except Exception as e:
                logging.warning(f"Failed to apply sample_id_map join: {str(e)}; using processed y.csv")
    
    # 加载PCA协变量
    pca_path = os.path.join(CONFIG["PROCESSED"], "pca_covariates.csv")
    logging.info(f"Loading PCA covariates from {pca_path}")
    pca_df = pd.read_csv(pca_path)
    
    # 处理第一行
    if 'POS' in pca_df.columns or 'REF' in pca_df.columns:
        pca_df = pca_df.iloc[1:].reset_index(drop=True)
    
    # 对齐样本
    logging.info("Aligning samples between datasets")
    
    # 创建共同的样本ID集合
    common_samples = list(set(X_df['Sample_ID']).intersection(set(y_df['Sample_ID'])).intersection(set(pca_df['Sample_ID'])))
    logging.info(f"Found {len(common_samples)} common samples across all datasets")
    
    # 过滤所有数据集以只包含共同样本
    X_filtered = X_df[X_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID').reset_index(drop=True)
    y_filtered = y_df[y_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID').reset_index(drop=True)
    pca_filtered = pca_df[pca_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID').reset_index(drop=True)
    
    # 验证对齐
    assert (X_filtered['Sample_ID'].values == y_filtered['Sample_ID'].values).all(), "Sample alignment failed between X and y"
    assert (X_filtered['Sample_ID'].values == pca_filtered['Sample_ID'].values).all(), "Sample alignment failed between X and PCA"
    
    # 对于ipk_out_raw，确保使用真实的数值型表型
    if CONFIG["DATASET"] == 'ipk_out_raw':
        # 检查TARGET_COLUMN是否存在且为数值型
        if CONFIG["TARGET_COLUMN"] not in y_filtered.columns:
            # 尝试查找合适的数值型目标列
            numeric_columns = y_filtered.select_dtypes(include=[np.number]).columns.tolist()
            # 排除ID列
            numeric_columns = [col for col in numeric_columns if 'ID' not in col.upper() and 'SAMPLE' not in col.upper()]
            
            if not numeric_columns:
                logging.error("No numeric phenotype columns found in y.csv for ipk_out_raw")
                sys.exit(1)
            
            # 使用第一个数值型列作为目标
            CONFIG["TARGET_COLUMN"] = numeric_columns[0]
            logging.info(f"Using {CONFIG['TARGET_COLUMN']} as target column for ipk_out_raw")
        
        # 验证目标列是数值型
        if not pd.api.types.is_numeric_dtype(y_filtered[CONFIG["TARGET_COLUMN"]]):
            logging.error(f"Target column {CONFIG['TARGET_COLUMN']} is not numeric")
            sys.exit(1)
    
    # 对于其他数据集，确保目标列存在
    elif CONFIG["TARGET_COLUMN"] not in y_filtered.columns:
        logging.error(f"Target column {CONFIG['TARGET_COLUMN']} not found in y.csv")
        sys.exit(1)
    
    # 提取特征（排除Sample_ID）
    X_features = X_filtered.drop('Sample_ID', axis=1)
    pca_features = pca_filtered.drop('Sample_ID', axis=1)
    
    # 合并X和PCA协变量
    X_combined = pd.concat([X_features, pca_features], axis=1)
    
    # 提取目标变量
    y = y_filtered[CONFIG["TARGET_COLUMN"]]
    
    # 检查目标变量中的NaN值
    nan_count = y.isna().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values in target variable. Removing these samples.")
        # 移除NaN目标值的样本
        valid_indices = ~y.isna()
        X_combined = X_combined[valid_indices].reset_index(drop=True)
        y = y[valid_indices].reset_index(drop=True)
        logging.info(f"After removing NaN targets: {X_combined.shape[0]} samples remaining")
    
    # 基础数据信息
    logging.info(f"Final data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    logging.info(f"Using target column: {CONFIG['TARGET_COLUMN']}")
    
    # 处理特征中的NaN/inf值
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
    X_combined = X_combined.fillna(X_combined.median())
    
    # NEW: Add scaling here if not already handled by models (some models need it)
    # Actually, we keep raw features for models that don't need scaling (trees)
    # and linear models will scale their own training set in train_models loop.
    
    # 转换为numpy数组
    X_array = X_combined.values.astype(float)
    y_array = y.values.astype(float)
    
    return X_array, y_array, X_filtered['Sample_ID'].tolist()

def handle_splits(sample_ids):
    """处理数据分割：如果splits.json存在则重用，否则创建70/15/15分割"""
    splits_path = os.path.join(CONFIG["OUTDIR"], "metrics", "splits.json")
    
    if os.path.exists(splits_path) and not CONFIG["OVERWRITE_PREVIOUS"]:
        logging.info(f"Loading existing splits from {splits_path}")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        return splits
    else:
        logging.info("Creating new splits with 70/15/15 ratio")
        
        # 创建训练集和剩余集（30%）
        train_idx, remaining_idx = train_test_split(
            range(len(sample_ids)), 
            test_size=0.3, 
            random_state=CONFIG["RANDOM_SEED"]
        )
        
        # 从剩余集中创建验证集和测试集（各15%）
        val_idx, test_idx = train_test_split(
            remaining_idx, 
            test_size=0.5, 
            random_state=CONFIG["RANDOM_SEED"]
        )
        
        splits = {
            "train": [sample_ids[i] for i in train_idx],
            "validation": [sample_ids[i] for i in val_idx],
            "test": [sample_ids[i] for i in test_idx],
            "seed": CONFIG["RANDOM_SEED"],
            "created_at": datetime.now().isoformat()
        }
        
        # 保存分割
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        # 同时保存到本次运行的日志中
        with open(os.path.join(CONFIG["OUTDIR"], "logs", f"splits_{CONFIG['RUN_ID']}.json"), 'w') as f:
            json.dump(splits, f, indent=2)
        
        logging.info(f"Created splits: Train={len(train_idx)}, Validation={len(val_idx)}, Test={len(test_idx)}")
        return splits

def feature_selection(X, y, model_type):
    """根据要求执行特征选择"""
    logging.info(f"Performing feature selection for {model_type} (original features: {X.shape[1]})")
    
    # 如果特征数超过10000，使用SelectKBest
    if X.shape[1] > 10000:
        # 对于树/boosting模型，使用k=1000或2000
        if model_type in ['randomforest', 'xgboost', 'lightgbm']:
            k = 2000  # 使用较大的值以保留更多信息
            logging.info(f"Using SelectKBest with f_regression for {model_type} (k={k})")
            selector = SelectKBest(f_regression, k=k)
            selector.fit(X, y)
            X_selected = selector.transform(X)
            feature_indices = selector.get_support(indices=True)
            selected_k = k
        else:
            # 对于其他模型，可能需要不同的策略
            X_selected = X
            feature_indices = np.arange(X.shape[1])
            selected_k = "all"
    else:
        # 对于小数据集，使用所有特征
        logging.info("Using all features for model (k=all)")
        X_selected = X
        feature_indices = np.arange(X.shape[1])
        selected_k = "all"
    
    logging.info(f"Feature selection completed, retained {X_selected.shape[1]} features")
    
    return X_selected, feature_indices, selected_k

def remove_constant_columns(X):
    """移除常量或全零列以提高数值稳定性"""
    # 计算每列的标准差
    std_devs = np.std(X, axis=0)
    
    # 找出非零标准差的列索引
    non_constant_indices = np.where(std_devs > 1e-10)[0]
    
    # 计算删除的列数
    removed_columns = X.shape[1] - len(non_constant_indices)
    
    logging.info(f"Removed {removed_columns} constant or all-zero columns")
    
    return X[:, non_constant_indices], non_constant_indices

def train_models(X, y):
    """使用嵌套交叉验证训练所有模型"""
    logging.info("Training models with nested cross-validation")
    
    # 定义外层和内层CV策略
    outer_cv = KFold(n_splits=CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    inner_cv = KFold(n_splits=CONFIG["INNER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    
    # Get model configurations from registry
    models = get_model_configs(CONFIG)
    
    logging.info(f"Will train the following models: {list(models.keys())}")
    
    # 加载现有结果
    all_metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", "all_models_metrics.json")
    results = {}
    if os.path.exists(all_metrics_path) and not CONFIG["OVERWRITE_PREVIOUS"]:
        try:
            with open(all_metrics_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                results = loaded
                logging.info("Loaded existing results")
            else:
                logging.warning("Existing results file is not a dict; starting fresh")
                results = {}
        except Exception as e:
            logging.warning(f"Failed to load existing results ({e}); starting fresh")
            results = {}
    
    # 训练每个模型
    for model_name, model_info in models.items():
        # 创建模型特定的子目录
        model_dir = os.path.join(CONFIG["OUTDIR"], "models", f"{model_name}_{CONFIG['RUN_ID']}")
        os.makedirs(model_dir, exist_ok=True)
        
        # 生成带run_id的指标文件路径
        metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", f"{model_name}_metrics_{CONFIG['RUN_ID']}.json")
        
        logging.info(f"Training {model_name} (run_id: {CONFIG['RUN_ID']})")
        start_time = time.time()
        
        try:
            # 初始化列表存储CV结果
            all_r2_scores = []
            all_rmse_scores = []
            all_mae_scores = []
            all_best_params = []
            
            # 特征选择
            if model_info['needs_feature_selection']:
                X_selected, feature_indices, selected_k = feature_selection(X, y, model_name)
                logging.info(f"Using selected features for {model_name}")
                joblib.dump(feature_indices, os.path.join(model_dir, f"feature_indices.joblib"))
            else:
                X_selected = X
                selected_k = "all"
            
            # 对于线性模型，移除常量列以提高数值稳定性
            original_feature_count = X_selected.shape[1]
            kept_indices = None
            if model_name in ['ridge', 'lasso', 'elasticnet']:
                X_selected, kept_indices = remove_constant_columns(X_selected)
                logging.info(f"{model_name}: Final feature count after removing constant columns: {X_selected.shape[1]}")
                if kept_indices is not None:
                    joblib.dump(kept_indices, os.path.join(model_dir, f"kept_indices.joblib"))
            
            # 嵌套CV循环
            for i, (train_idx, test_idx) in enumerate(outer_cv.split(X_selected)):
                logging.info(f"Outer fold {i+1}/{CONFIG['OUTER_CV_FOLDS']}")
                
                # 分割数据
                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 应用缩放（如果需要）
                if model_info['needs_scaling']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    joblib.dump(scaler, os.path.join(model_dir, f"scaler_fold_{i+1}.joblib"))
                
                # 内层CV用于超参数调优
                if model_info.get('early_stopping', False):
                    # 对于支持早停的模型，我们需要创建验证集
                    logging.info(f"Using early stopping with patience=50 for {model_name}")
                    
                    # 为早停创建验证集
                    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                        X_train, y_train, test_size=0.1, random_state=CONFIG["RANDOM_SEED"]
                    )
                    
                    if model_name == 'xgboost' and XGBOOST_AVAILABLE:
                        grid = GridSearchCV(
                            model_info['estimator'],
                            model_info['params'],
                            cv=inner_cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            fit_params={
                                'early_stopping_rounds': 50,
                                'eval_metric': 'rmse',
                                'eval_set': [(X_val_inner, y_val_inner)],
                                'verbose': 0
                            }
                        )
                    elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                        grid = GridSearchCV(
                            model_info['estimator'],
                            model_info['params'],
                            cv=inner_cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            fit_params={
                                'early_stopping_rounds': 50,
                                'eval_metric': 'rmse',
                                'eval_set': [(X_val_inner, y_val_inner)],
                                'verbose': 0
                            }
                        )
                    else:
                        grid = GridSearchCV(
                            model_info['estimator'],
                            model_info['params'],
                            cv=inner_cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1
                        )
                else:
                    grid = GridSearchCV(
                        model_info['estimator'],
                        model_info['params'],
                        cv=inner_cv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                
                grid.fit(X_train, y_train)
                
                # 在测试折上预测
                y_pred = grid.best_estimator_.predict(X_test)
                
                # 计算指标
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # 存储结果
                all_r2_scores.append(r2)
                all_rmse_scores.append(rmse)
                all_mae_scores.append(mae)
                all_best_params.append(grid.best_params_)
                
                logging.info(f"  Fold {i+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            # 在所有数据上训练最终模型
            if model_info['needs_scaling']:
                final_scaler = StandardScaler()
                X_final = final_scaler.fit_transform(X_selected)
                joblib.dump(final_scaler, os.path.join(model_dir, "final_scaler.joblib"))
            else:
                X_final = X_selected
            
            # 训练最终模型
            if model_info.get('early_stopping', False):
                # 为早停创建验证集
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                    X_final, y, test_size=0.1, random_state=CONFIG["RANDOM_SEED"]
                )
                
                final_grid = GridSearchCV(
                    model_info['estimator'],
                    model_info['params'],
                    cv=inner_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                # 对于支持早停的模型，需要特殊处理
                if model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    final_grid = GridSearchCV(
                        model_info['estimator'],
                        model_info['params'],
                        cv=inner_cv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        fit_params={
                            'early_stopping_rounds': 50,
                            'eval_metric': 'rmse',
                            'eval_set': [(X_val_final, y_val_final)],
                            'verbose': 0
                        }
                    )
                elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    final_grid = GridSearchCV(
                        model_info['estimator'],
                        model_info['params'],
                        cv=inner_cv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        fit_params={
                            'early_stopping_rounds': 50,
                            'eval_metric': 'rmse',
                            'eval_set': [(X_val_final, y_val_final)],
                            'verbose': 0
                        }
                    )
            else:
                final_grid = GridSearchCV(
                    model_info['estimator'],
                    model_info['params'],
                    cv=inner_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            
            final_grid.fit(X_final, y)
            
            # 保存最终模型
            joblib.dump(final_grid.best_estimator_, os.path.join(model_dir, "final_model.joblib"))
            
            # 汇总结果
            model_results = {
                'cv_r2_mean': float(np.mean(all_r2_scores)),
                'cv_r2_std': float(np.std(all_r2_scores)),
                'cv_rmse_mean': float(np.mean(all_rmse_scores)),
                'cv_rmse_std': float(np.std(all_rmse_scores)),
                'cv_mae_mean': float(np.mean(all_mae_scores)),
                'cv_mae_std': float(np.std(all_mae_scores)),
                'all_fold_params': all_best_params,
                'final_best_params': final_grid.best_params_,
                'train_time': float(time.time() - start_time),
                'features_count': X_selected.shape[1],
                'selected_k': selected_k,
                'run_id': CONFIG["RUN_ID"],
                'run_type': CONFIG["RUN_TYPE"],
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存模型特定的指标
            with open(metrics_path, "w") as f:
                json.dump(model_results, f, indent=2)
            
            # 更新总体结果
            # 为了避免覆盖，我们将使用run_id作为键的一部分
            results_key = f"{model_name}_{CONFIG['RUN_ID']}"
            results[results_key] = model_results
            
            logging.info(f"Completed training {model_name} in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error with {model_name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # 记录错误但继续处理其他模型
            error_results = {
                "error": str(e),
                "train_time": time.time() - start_time,
                "run_id": CONFIG["RUN_ID"],
                "timestamp": datetime.now().isoformat()
            }
            results[f"{model_name}_{CONFIG['RUN_ID']}"] = error_results
            
            with open(metrics_path, "w") as f:
                json.dump(error_results, f, indent=2)
    
    # 保存所有指标
    with open(all_metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def update_summary(results):
    """更新摘要报告"""
    logging.info("Updating summary report")
    
    summary_file = os.path.join(CONFIG["OUTDIR"], "summary.md")
    
    # 过滤出本次运行的结果
    current_run_results = {}
    for key, value in results.items():
        if CONFIG['RUN_ID'] in key and 'error' not in value:
            # 提取模型名称（去掉run_id部分）
            model_name = key.replace(f"_{CONFIG['RUN_ID']}", '')
            current_run_results[model_name] = value
    
    # 按R2排序
    sorted_models = sorted(current_run_results.items(), 
                          key=lambda x: x[1].get('cv_r2_mean', -float('inf')), 
                          reverse=True)
    
    # 如果文件存在，读取现有内容
    existing_content = ""
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding="utf-8") as f:
            existing_content = f.read()
    
    # 写入新的内容（添加到开头）
    with open(summary_file, "w", encoding="utf-8") as f:
        # 写入本次运行的标题
        f.write(f"# {CONFIG['DATASET']} Modeling Results - Rerun ({CONFIG['RUN_ID']})\n\n")
        
        # 配置信息
        f.write("## Configuration\n\n")
        for key, value in CONFIG.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        # 交叉验证结果
        f.write("## Cross-Validation Results\n\n")
        f.write("| Model | CV R2 Mean | CV R2 Std | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV MAE Std | Features Count | Selected K |\n")
        f.write("|-------|------------|-----------|--------------|-------------|-------------|------------|----------------|------------|\n")
        
        for model_name, metrics in sorted_models:
            f.write(f"| {model_name} | {metrics['cv_r2_mean']:.4f} | {metrics['cv_r2_std']:.4f} | {metrics['cv_rmse_mean']:.4f} | {metrics['cv_rmse_std']:.4f} | {metrics['cv_mae_mean']:.4f} | {metrics['cv_mae_std']:.4f} | {metrics['features_count']} | {metrics['selected_k']} |\n")
        
        # 超参数网格信息
        f.write("\n## Hyperparameter Grids\n\n")
        for model_name, metrics in sorted_models:
            f.write(f"### {model_name}\n\n")
            if 'final_best_params' in metrics:
                f.write("Best parameters:\n\n")
                for param, value in metrics['final_best_params'].items():
                    f.write(f"- **{param}**: {value}\n")
                f.write("\n")
        
        # 写入原有的内容
        if existing_content:
            f.write("\n---\n\n")
            f.write("## Previous Runs\n\n")
            f.write(existing_content)

def generate_comparison_table(results):
    """生成比较表格"""
    logging.info("Generating comparison table")
    
    # 过滤出本次运行的结果
    current_run_results = {}
    for key, value in results.items():
        if CONFIG['RUN_ID'] in key and 'error' not in value:
            model_name = key.replace(f"_{CONFIG['RUN_ID']}", '')
            current_run_results[model_name] = value
    
    # 按R2排序
    sorted_models = sorted(current_run_results.items(), 
                          key=lambda x: x[1].get('cv_r2_mean', -float('inf')), 
                          reverse=True)
    
    # 创建比较表格文件
    comparison_file = os.path.join(CONFIG["OUTDIR"], "metrics", f"comparison_table_{CONFIG['RUN_ID']}.md")
    
    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write(f"# {CONFIG['DATASET']} Model Comparison ({CONFIG['RUN_ID']})\n\n")
        
        f.write("## R² Mean ± Std\n\n")
        f.write("| Model | R² Mean ± Std | RMSE | MAE |\n")
        f.write("|-------|--------------|------|-----|\n")
        
        for model_name, metrics in sorted_models:
            r2_str = f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
            f.write(f"| {model_name} | {r2_str} | {metrics['cv_rmse_mean']:.4f} | {metrics['cv_mae_mean']:.4f} |\n")
        
        # 确定最佳模型
        if sorted_models:
            best_model = sorted_models[0]
            f.write("\n## Winner Per Dataset\n\n")
            f.write(f"**Best model for {CONFIG['DATASET']}: {best_model[0]}** with R² = {best_model[1]['cv_r2_mean']:.4f}\n\n")
            f.write("### Best Parameters:\n\n")
            for param, value in best_model[1]['final_best_params'].items():
                f.write(f"- **{param}**: {value}\n")
    
    logging.info(f"Comparison table saved to {comparison_file}")
    return comparison_file

def main():
    """主函数"""
    logging.info(f"Starting unified modeling pipeline for {CONFIG['DATASET']}")
    
    # 记录环境
    log_environment()
    
    # 执行飞行前检查
    if not preflight_checks():
        logging.error("Pre-flight checks failed, exiting")
        sys.exit(1)
    
    # 加载并对齐数据
    X, y, sample_ids = load_and_align_data()
    
    # 处理数据分割
    splits = handle_splits(sample_ids)
    
    # 训练模型
    results = train_models(X, y)
    
    # 更新摘要
    update_summary(results)
    
    # 生成比较表格
    comparison_file = generate_comparison_table(results)
    
    logging.info(f"Pipeline completed successfully for {CONFIG['DATASET']}")
    logging.info(f"Comparison table available at {comparison_file}")

if __name__ == "__main__":
    main()