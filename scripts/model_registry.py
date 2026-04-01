# Model registry and hooks for future models
"""
A central registry for managing models in the modeling pipelines.
This provides a clean interface for adding new models without modifying the core pipeline code.
"""

import logging
import importlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global model registry
_model_registry = {}  # type: Dict[str, Dict[str, Any]]

def register_model(
    name: str,
    estimator_class: Any,
    param_grid: Dict[str, Any],
    needs_scaling: bool = False,
    needs_feature_selection: bool = False,
    early_stopping: bool = False,
    dependencies: Optional[Dict[str, str]] = None
):
    """
    Register a new model in the registry.
    
    Args:
        name: Unique name for the model
        estimator_class: The model class (not instantiated)
        param_grid: Hyperparameter grid for grid search
        needs_scaling: Whether the model requires feature scaling
        needs_feature_selection: Whether the model requires feature selection
        early_stopping: Whether the model supports early stopping
        dependencies: Dictionary of required packages and their versions
    
    Returns:
        True if registration was successful, False otherwise
    """
    if name in _model_registry:
        logger.warning(f"Model '{name}' already registered, skipping")
        return False
    
    _model_registry[name] = {
        'estimator_class': estimator_class,
        'param_grid': param_grid,
        'needs_scaling': needs_scaling,
        'needs_feature_selection': needs_feature_selection,
        'early_stopping': early_stopping,
        'dependencies': dependencies or {}
    }
    
    logger.info(f"Model '{name}' registered successfully")
    return True

def get_model_config(name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a registered model.
    
    Args:
        name: Name of the registered model
    
    Returns:
        Model configuration dictionary if found, None otherwise
    """
    return _model_registry.get(name)

def list_registered_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered models.
    
    Returns:
        Dictionary of all registered models
    """
    return _model_registry.copy()

def autoload_models():
    """
    Automatically load and register built-in models.
    This function tries to import popular models and register them if available.
    """
    global _model_registry
    
    # Clear registry before autoloading
    _model_registry = {}
    
    # 1. Sklearn built-in models
    try:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        
        # Register Ridge
        register_model(
            'ridge',
            Ridge,
            {'alpha': [1e-6, 1e-4, 1e-2, 1, 10, 100]},
            needs_scaling=True,
            needs_feature_selection=False,
            early_stopping=False,
            dependencies={'scikit-learn': '>=1.0'}
        )
        
        # Register Lasso
        register_model(
            'lasso',
            Lasso,
            {'alpha': [1e-6, 1e-4, 1e-2, 1, 10, 100], 'max_iter': [10000]},
            needs_scaling=True,
            needs_feature_selection=False,
            early_stopping=False,
            dependencies={'scikit-learn': '>=1.0'}
        )
        
        # Register ElasticNet
        register_model(
            'elasticnet',
            ElasticNet,
            {'alpha': [1e-6, 1e-4, 1e-2, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9], 'max_iter': [10000]},
            needs_scaling=True,
            needs_feature_selection=False,
            early_stopping=False,
            dependencies={'scikit-learn': '>=1.0'}
        )
        
        # Register Random Forest
        register_model(
            'randomforest',
            RandomForestRegressor,
            {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'max_features': ['auto', 'sqrt']},
            needs_scaling=False,
            needs_feature_selection=True,
            early_stopping=False,
            dependencies={'scikit-learn': '>=1.0'}
        )
        
        # Register SVR
        register_model(
            'svr',
            SVR,
            {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1], 'kernel': ['linear', 'rbf'], 'gamma': ['scale']},
            needs_scaling=True,
            needs_feature_selection=True,
            early_stopping=False,
            dependencies={'scikit-learn': '>=1.0'}
        )
        
        logger.info("Sklearn built-in models registered successfully")
    except ImportError as e:
        logger.warning(f"Failed to register sklearn models: {e}")
    
    # 2. XGBoost (if available)
    try:
        from xgboost import XGBRegressor
        register_model(
            'xgboost',
            XGBRegressor,
            {
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_lambda': [1, 5],
                'tree_method': ['hist']
            },
            needs_scaling=False,
            needs_feature_selection=True,
            early_stopping=True,
            dependencies={'xgboost': '>=1.5'}
        )
        logger.info("XGBoost registered successfully")
    except ImportError as e:
        logger.warning(f"Failed to register XGBoost: {e}")
    
    # 3. LightGBM (if available)
    try:
        from lightgbm import LGBMRegressor
        register_model(
            'lightgbm',
            LGBMRegressor,
            {
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'num_leaves': [31, 63],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            needs_scaling=False,
            needs_feature_selection=True,
            early_stopping=True,
            dependencies={'lightgbm': '>=3.3'}
        )
        logger.info("LightGBM registered successfully")
    except ImportError as e:
        logger.warning(f"Failed to register LightGBM: {e}")
    
    logger.info(f"Total registered models: {len(_model_registry)}")

def load_external_model_config(config_path: str):
    """
    Load external model configurations from a Python file.
    The file should contain a function `register_models()` that registers models.
    
    Args:
        config_path: Path to the external configuration file
    """
    try:
        # Add the directory containing the config file to the path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(config_path))
        
        # Import the config module
        module_name = os.path.basename(config_path).replace('.py', '')
        config_module = importlib.import_module(module_name)
        
        # Call register_models() if it exists
        if hasattr(config_module, 'register_models'):
            config_module.register_models()
            logger.info(f"Successfully loaded external models from {config_path}")
        else:
            logger.warning(f"No 'register_models' function found in {config_path}")
            
    except Exception as e:
        logger.error(f"Failed to load external model config: {e}")

def get_model_by_name(name: str, **kwargs) -> Optional[Any]:
    """
    Get an instantiated model by name from the registry.
    
    Args:
        name: Name of the registered model
        **kwargs: Additional parameters to pass to the estimator constructor
        
    Returns:
        Instantiated model if found, None otherwise
    """
    model_config = _model_registry.get(name)
    if not model_config:
        logger.error(f"Model '{name}' not found in registry")
        return None
    
    # Add random_state if available
    if 'random_state' not in kwargs:
        # Try to import CONFIG if available
        try:
            from scripts.unified_modeling_pipeline import CONFIG
            kwargs['random_state'] = CONFIG.get('RANDOM_SEED', 42)
        except ImportError:
            kwargs['random_state'] = 42
    
    # Add verbosity settings for XGBoost and LightGBM
    if name == 'xgboost' and 'verbosity' not in kwargs:
        kwargs['verbosity'] = 0
    elif name == 'lightgbm' and 'verbosity' not in kwargs:
        kwargs['verbosity'] = -1
    
    try:
        return model_config['estimator_class'](**kwargs)
    except Exception as e:
        logger.error(f"Failed to instantiate model '{name}': {e}")
        return None

def get_model_configs(config=None):
    """
    Get all registered models formatted for the pipeline.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Dictionary of model configurations
    """
    models = {}
    for name, info in _model_registry.items():
        models[name] = {
            'estimator': info['estimator_class'](),
            'params': info['param_grid'],
            'needs_scaling': info['needs_scaling'],
            'needs_feature_selection': info['needs_feature_selection'],
            'early_stopping': info['early_stopping']
        }
    return models

# Initialize registry on module import
autoload_models()