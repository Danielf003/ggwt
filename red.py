import pandas as pd
import numpy as np
import requests
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

# =============================================================================
# КОНФИГ
# =============================================================================
TRAIN_PATH = '../ggwt_data/train_dataset.csv'
VALID_PATH = '../ggwt_data/valid_features.csv'
OUTPUT_VALID = 'solverdata.csv'

TARGET = 'Выработка. Результирующий расчет'
DT_COL = 'METEOFORECASTHOUR_OPENM_Datetime'

CAPACITY    = 90.09
N_TURBINES  = 26
LATITUDE    = 46.8268455973
LONGITUDE   = 38.7179393185

# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# =============================================================================
# Модель считается активной, если список сидов не пустой
MODEL_CONFIGS = {
    'LightGBM': {
        'seeds': [42],#[42, 123, 777, 2024, 9001],
        'weight': None,  # None = будет подобран автоматически
        'params': {
            'objective': 'quantile',
            'alpha': 0.5,
            'n_jobs': -1,
            'verbose': -1
        },
    },
    'CatBoost': {
        'seeds': [],#[42, 123, 777, 2024, 9001],
        'weight': None,
        'params': {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 0.1,
            'border_count': 128,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'verbose': False
        },
    },
    'XGBoost': {
        'seeds': [],#[42, 123, 777, 2024, 9001],
        'weight': None,
        'params': {
            'n_estimators': 1000,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'verbosity': 0
        },
    }
}

LGB_ITER_MULTIPLIER = 1.1
N_WAKE_SECTORS = 72

# =============================================================================
# АБСТРАКТНЫЙ БЕЙС-КЛАСС ДЛЯ МОДЕЛЕЙ
# =============================================================================
class BaseModelWrapper(ABC):
    """Абстрактный класс-обертка для ML моделей с единым интерфейсом"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.models = []  # список обученных моделей (ансамбль)
        self.is_active = len(config.get('seeds', [])) > 0
        self.weight = config.get('weight')
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_holdout: Optional[np.ndarray] = None, 
                       y_holdout: Optional[np.ndarray] = None) -> List:
        """
        Обучает ансамбль моделей для каждого seed.
        Если передан holdout, использует раннюю остановку.
        Возвращает список обученных моделей.
        """
        if not self.is_active:
            print(f"⚠️ Модель {self.name} отключена (пустой список seeds)")
            return []
        
        seeds = self.config['seeds']
        models = []
        
        for seed in seeds:
            if X_holdout is not None and y_holdout is not None:
                # Обучаем с валидацией
                model = self.fit_with_early_stopping(seed, X_train, y_train, X_holdout, y_holdout)
            else:
                # Финальное обучение на всех данных без валидации
                model = self.fit_final(seed, X_train, y_train)
            
            models.append(model)
        
        self.models = models
        return models
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Усредненный прогноз ансамбля"""
        if not self.models:
            return np.zeros(len(X))
        
        predictions = [np.clip(model.predict(X), 0, CAPACITY) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def predict_holdout(self, X_ho: np.ndarray) -> np.ndarray:
        """Прогноз на holdout для подбора весов"""
        if not self.models:
            return np.zeros(len(X_ho))
        
        predictions = [np.clip(model.predict(X_ho), 0, CAPACITY) for model in self.models]
        return np.mean(predictions, axis=0)
    
    @abstractmethod
    def fit_with_early_stopping(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray):
        """Обучает модель с ранней остановкой"""
        pass
    
    @abstractmethod
    def fit_final(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray):
        """Финальное обучение без ранней остановки"""
        pass

# =============================================================================
# РЕАЛИЗАЦИИ КОНКРЕТНЫХ МОДЕЛЕЙ
# =============================================================================
class LightGBMWrapper(BaseModelWrapper):
    def fit_with_early_stopping(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray):
        params = self.config['params'].copy()
        params['random_state'] = seed
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(75, verbose=False)])
        return model
    
    def fit_final(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray):
        params = self.config['params'].copy()
        params['random_state'] = seed
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        return model


class CatBoostWrapper(BaseModelWrapper):
    def fit_with_early_stopping(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray):
        params = self.config['params'].copy()
        params['random_seed'] = seed
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                  early_stopping_rounds=30, verbose=False)
        return model
    
    def fit_final(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray):
        params = self.config['params'].copy()
        params['random_seed'] = seed
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        return model


class XGBoostWrapper(BaseModelWrapper):
    def fit_with_early_stopping(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray):
        params = self.config['params'].copy()
        params['random_state'] = seed
        params['early_stopping_rounds'] = 30
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model
    
    def fit_final(self, seed: int, X_tr: np.ndarray, y_tr: np.ndarray):
        params = self.config['params'].copy()
        params['random_state'] = seed
        # # Убираем early_stopping_rounds, если он есть
        # params.pop('early_stopping_rounds', None)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        return model

# =============================================================================
# ФАБРИКА МОДЕЛЕЙ
# =============================================================================
def create_model_wrappers() -> Dict[str, BaseModelWrapper]:
    """Создает обертки для всех моделей из конфига"""
    wrappers = {
        'LightGBM': LightGBMWrapper('LightGBM', MODEL_CONFIGS['LightGBM']),
        'CatBoost': CatBoostWrapper('CatBoost', MODEL_CONFIGS['CatBoost']),
        'XGBoost': XGBoostWrapper('XGBoost', MODEL_CONFIGS['XGBoost'])
    }
    return wrappers

# =============================================================================
# 1-11. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ (БЕЗ ИЗМЕНЕНИЙ)
# =============================================================================
# ... (весь код загрузки и обработки данных остается без изменений до раздела 12)
# [Сохранена оригинальная логика загрузки данных, так как она не требует изменений]

# =============================================================================
# 12. ОПТИМИЗАЦИЯ LightGBM (опционально)
# =============================================================================
def optimize_lightgbm(X: np.ndarray, y: np.ndarray) -> Dict:
    """Оптимизация гиперпараметров LightGBM через Optuna"""
    print("Оптимизация LightGBM...")
    
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'objective': 'quantile',
            'alpha': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_val), 0, CAPACITY)
            scores.append(mean_absolute_error(y_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lgb, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'quantile',
        'alpha': 0.5,
        'n_jobs': -1,
        'verbose': -1
    })
    print(f"Лучшие параметры LightGBM: {best_params}")
    return best_params

def optimize_catboost(X: np.ndarray, y: np.ndarray) -> Dict:
    """Оптимизация гиперпараметров CatBoost через Optuna"""
    print("=" * 60)
    print("Оптимизация CatBoost...")
    print("=" * 60)
    
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'verbose': False
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = np.clip(model.predict(X_val), 0, CAPACITY)
            scores.append(mean_absolute_error(y_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_cat, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'verbose': False
    })
    
    print("\n" + "=" * 60)
    print("Лучшие параметры CatBoost (скопируйте и вставьте в конфиг):")
    print("=" * 60)
    print(f"best_cat_params = {best_params}")
    print("=" * 60 + "\n")
    
    return best_params

def optimize_xgboost(X: np.ndarray, y: np.ndarray) -> Dict:
    """Оптимизация гиперпараметров XGBoost через Optuna"""
    print("=" * 60)
    print("Оптимизация XGBoost...")
    print("=" * 60)
    
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'random_state': 42,
            'verbosity': 0
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = np.clip(model.predict(X_val), 0, CAPACITY)
            scores.append(mean_absolute_error(y_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.5,
        'verbosity': 0
    })
    
    print("\n" + "=" * 60)
    print("Лучшие параметры XGBoost (скопируйте и вставьте в конфиг):")
    print("=" * 60)
    print(f"best_xgb_params = {best_params}")
    print("=" * 60 + "\n")
    
    return best_params

# =============================================================================
# 13. HOLDOUT-ВАЛИДАЦИЯ + ПОДБОР ВЕСОВ
# =============================================================================
def train_and_optimize_weights(X_tr: np.ndarray, y_tr: np.ndarray, 
                               X_ho: np.ndarray, y_ho: np.ndarray,
                               model_wrappers: Dict[str, BaseModelWrapper]) -> Tuple[Dict[str, float], Dict[str, BaseModelWrapper]]:
    """
    Обучает модели на тренировочных данных с валидацией на holdout,
    подбирает оптимальные веса для активных моделей.
    """
    active_models = {name: wrapper for name, wrapper in model_wrappers.items() if wrapper.is_active}
    
    if not active_models:
        raise ValueError("Нет активных моделей для обучения!")
    
    print(f"\nАктивные модели для обучения: {list(active_models.keys())}")
    
    holdout_predictions = {}
    
    for name, wrapper in active_models.items():
        print(f"Обучение {name} на train с валидацией на holdout...")
        wrapper.train_ensemble(X_tr, y_tr, X_ho, y_ho)
        holdout_predictions[name] = wrapper.predict_holdout(X_ho)
    
    # Подбор весов для активных моделей
    model_names = list(active_models.keys())
    n_models = len(model_names)
    best_weights = {name: 0.0 for name in model_names}
    best_mae = np.inf
    
    # Генерируем все возможные комбинации весов с шагом 0.05
    def generate_weights(current_weights: List[float], remaining_idx: int, remaining_sum: float):
        nonlocal best_weights, best_mae
        
        if remaining_idx == n_models - 1:
            final_weights = current_weights + [remaining_sum]
            if all(w >= 0 for w in final_weights):
                pred = np.zeros(len(y_ho))
                for idx, name in enumerate(model_names):
                    pred += final_weights[idx] * holdout_predictions[name]
                
                mae = mean_absolute_error(y_ho, pred)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = {name: final_weights[idx] for idx, name in enumerate(model_names)}
        else:
            for i in range(21):  # 0.0, 0.05, ..., 1.0
                w = i / 20.0
                if w <= remaining_sum:
                    generate_weights(current_weights + [w], remaining_idx + 1, remaining_sum - w)
    
    generate_weights([], 0, 1.0)
    
    print(f"\nОптимальные веса (MAE={best_mae:.3f} МВт):")
    for name, weight in best_weights.items():
        print(f"  {name}: {weight:.2f}")
    
    return best_weights, model_wrappers

# =============================================================================
# 14. ФИНАЛЬНОЕ ОБУЧЕНИЕ ТОЛЬКО ДЛЯ МОДЕЛЕЙ С НЕНУЛЕВЫМ ВЕСОМ
# =============================================================================
def final_train_and_predict(X: np.ndarray, y: np.ndarray, 
                           X_valid: np.ndarray,
                           model_wrappers: Dict[str, BaseModelWrapper],
                           weights: Dict[str, float],
                           optimize_lgbm_iters=False) -> np.ndarray:
    """
    Финальное обучение только тех моделей, у которых вес > 0
    """
    active_models = {name: wrapper for name, wrapper in model_wrappers.items() 
                    if wrapper.is_active and weights.get(name, 0) > 0}
    
    if not active_models:
        raise ValueError("Нет моделей с положительным весом для финального обучения!")
    
    print(f"\nМодели для финального обучения (вес > 0): {list(active_models.keys())}")
    
    # Специальная обработка для LightGBM: определяем оптимальное количество итераций
    if optimize_lgbm_iters and 'LightGBM' in active_models and MODEL_CONFIGS['LightGBM']['seeds']:
        lgb_wrapper = active_models['LightGBM']
        if lgb_wrapper.models:
            n_iter_ho = np.median([getattr(m, 'best_iteration_', 1500) for m in lgb_wrapper.models]) 
            n_iter_ho = int(n_iter_ho * LGB_ITER_MULTIPLIER)
            MODEL_CONFIGS['LightGBM']['params']['n_estimators'] = n_iter_ho
            # Пересоздаем обертку с новыми параметрами
            active_models['LightGBM'] = LightGBMWrapper('LightGBM', MODEL_CONFIGS['LightGBM'])
    
    final_predictions = {}
    for name, wrapper in active_models.items():
        print(f"Финальное обучение {name} на всех данных...")
        wrapper.train_ensemble(X, y, None, None)
        final_predictions[name] = wrapper.predict(X_valid)
    
    # Комбинируем прогнозы с весами
    final_pred = np.zeros(len(X_valid))
    total_weight = sum(weights.get(name, 0) for name in final_predictions.keys())
    
    for name, pred in final_predictions.items():
        weight = weights.get(name, 0)
        if weight > 0:
            final_pred += (weight / total_weight) * pred
    
    return final_pred

# =============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# =============================================================================
def main():
    # [Здесь должен быть весь код загрузки и обработки данных]
    # Для краткости предполагаем, что переменные X, y, X_valid, valid_f уже созданы
    # (в реальном коде сюда вставляются все разделы 1-11)
    # =============================================================================
    # 1. ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ
    # =============================================================================
    train = pd.read_csv(TRAIN_PATH)
    valid = pd.read_csv(VALID_PATH)
    train[DT_COL] = pd.to_datetime(train[DT_COL])
    valid[DT_COL] = pd.to_datetime(valid[DT_COL])
    valid_order = valid[DT_COL].copy()
    train = train.sort_values(DT_COL).reset_index(drop=True)

    # =============================================================================
    # 2. ЗАГРУЗКА ERA5
    # =============================================================================
    def fetch_era5(start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": start_date, "end_date": end_date,
            "hourly": [
                "wind_speed_10m", "wind_speed_100m",
                "wind_direction_10m", "wind_direction_100m",
                "temperature_2m", "pressure_msl",
                #
                # "relative_humidity_2m",   
                # "dew_point_2m", # точка росы
            ],
            "timezone": "UTC"
        }
        r = requests.get(url, params=params).json()["hourly"]
        return pd.DataFrame({
            DT_COL: pd.to_datetime(r["time"]),
            "ws10_era5": r["wind_speed_10m"],
            "ws100_era5": r["wind_speed_100m"],
            "wd10_era5": r["wind_direction_10m"],
            "wd100_era5": r["wind_direction_100m"],
            "temp2m_era5": r["temperature_2m"],
            "pressure_era5": r["pressure_msl"],
            #
            # "rh2m_era5": r["relative_humidity_2m"],
            # "dewpoint2m_era5": r["dew_point_2m"],
        })

    print("Загрузка ERA5...")
    train_era5 = fetch_era5(train[DT_COL].min().strftime('%Y-%m-%d'),
                            train[DT_COL].max().strftime('%Y-%m-%d'))
    valid_era5 = fetch_era5(valid[DT_COL].min().strftime('%Y-%m-%d'),
                            valid[DT_COL].max().strftime('%Y-%m-%d'))

    # =============================================================================
    # 3. ЗАГРУЗКА NASA POWER
    # =============================================================================
    def fetch_nasa_power(start_date, end_date):
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            "parameters": "WS10M,WS50M,WS2M,RH2M,T2M,PS,QV2M,PRECTOTCORR",
            "community": "RE",
            "longitude": LONGITUDE,
            "latitude": LATITUDE,
            "start": start_date.replace('-', ''),
            "end": end_date.replace('-', ''),
            "format": "JSON"
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        rec = data["properties"]["parameter"]
        timestamps = sorted(rec["WS10M"].keys())
        return pd.DataFrame({
            DT_COL: pd.to_datetime(timestamps, format='%Y%m%d%H'),
            "ws10_power": [rec["WS10M"][t] for t in timestamps],
            "ws50_power": [rec["WS50M"][t] for t in timestamps],
            "ws2_power":  [rec["WS2M"][t] for t in timestamps],
            "rh2m_power": [rec["RH2M"][t] for t in timestamps],
            "temp2m_power": [rec["T2M"][t] for t in timestamps],
            "pressure_power": [rec["PS"][t] for t in timestamps],
            "qv2m_power": [rec["QV2M"][t] for t in timestamps],
            "precip_power": [rec["PRECTOTCORR"][t] for t in timestamps],
        })

    print("Загрузка NASA POWER...")
    train_power = fetch_nasa_power(train[DT_COL].min().strftime('%Y-%m-%d'),
                                train[DT_COL].max().strftime('%Y-%m-%d'))
    valid_power = fetch_nasa_power(valid[DT_COL].min().strftime('%Y-%m-%d'),
                                valid[DT_COL].max().strftime('%Y-%m-%d'))

    # =============================================================================
    # 4. ЗАГРУЗКА AIR QUALITY
    # =============================================================================
    def fetch_air_quality(start_date, end_date):
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "pm2_5,pm10,carbon_monoxide,dust",
            "timezone": "UTC"
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "hourly" in data and "time" in data["hourly"]:
                hourly = data["hourly"]
                return pd.DataFrame({
                    DT_COL: pd.to_datetime(hourly["time"]),
                    "pm2_5": hourly.get("pm2_5", None),
                    "pm10": hourly.get("pm10", None),
                    "co": hourly.get("carbon_monoxide", None),
                    "aod": hourly.get("dust", None)
                })
            else:
                return pd.DataFrame(columns=[DT_COL, "pm2_5", "pm10", "co", "aod"])
        except Exception as e:
            print(f"⚠️ Air Quality не загружен: {e}")
            return pd.DataFrame(columns=[DT_COL, "pm2_5", "pm10", "co", "aod"])

    print("Загрузка Air Quality...")
    train_aq = fetch_air_quality(train[DT_COL].min().strftime('%Y-%m-%d'),
                                train[DT_COL].max().strftime('%Y-%m-%d'))
    valid_aq = fetch_air_quality(valid[DT_COL].min().strftime('%Y-%m-%d'),
                                valid[DT_COL].max().strftime('%Y-%m-%d'))

    # =============================================================================
    # 5. РЕЛЬЕФ
    # =============================================================================
    def get_elevation(lat, lon):
        url = f"https://api.opentopodata.org/v1/aster30m"
        params = {"locations": f"{lat},{lon}"}
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                return r.json()["results"][0]["elevation"]
        except:
            pass
        return None

    def get_surrounding_elevations(lat, lon, radius_deg=0.01):
        points = {
            'center': (lat, lon),
            'north': (lat + radius_deg, lon),
            'south': (lat - radius_deg, lon),
            'east': (lat, lon + radius_deg),
            'west': (lat, lon - radius_deg),
        }
        elevs = {}
        for name, (la, lo) in points.items():
            e = get_elevation(la, lo)
            if e is not None:
                elevs[name] = e
        return elevs

    print("Получение высот вокруг станции...")
    elev_data = get_surrounding_elevations(LATITUDE, LONGITUDE)

    if len(elev_data) >= 3:
        heights = list(elev_data.values())
        median_h = np.median(heights)
        range_h = np.max(heights) - np.min(heights)
        std_h = np.std(heights)
        center_h = elev_data.get('center', median_h)
        relative_h = center_h - median_h

        slope_steepness = 0.0
        slope_aspect = 0.0
        radius_deg = 0.01
        if 'north' in elev_data and 'south' in elev_data:
            d_ns = (elev_data['north'] - elev_data['south']) / (2 * radius_deg * 111000)
        else:
            d_ns = 0.0
        if 'east' in elev_data and 'west' in elev_data:
            d_ew = (elev_data['east'] - elev_data['west']) / (2 * radius_deg * 111000 * np.cos(np.deg2rad(LATITUDE)))
        else:
            d_ew = 0.0
        slope_steepness = np.sqrt(d_ns**2 + d_ew**2)
        slope_aspect = np.arctan2(d_ns, d_ew)

        if slope_steepness < 0.02:
            terrain_type = 0
        elif slope_steepness < 0.1:
            terrain_type = 1
        else:
            terrain_type = 2

        print(f"Рельеф: median_h={median_h:.1f}, range={range_h:.1f}, std={std_h:.1f}, "
            f"relative={relative_h:.1f}, slope={slope_steepness:.4f}, aspect={np.rad2deg(slope_aspect):.1f}°, terrain={terrain_type}")
    else:
        median_h = 36.0; range_h = 0.0; std_h = 0.0; relative_h = 0.0
        slope_steepness = 0.0; slope_aspect = 0.0; terrain_type = 0
        print("Недостаточно данных рельефа, используем заглушки.")

    # =============================================================================
    # 6. ОБРАБОТКА NASA POWER
    # =============================================================================
    def process_power(df_power):
        ws10 = df_power['ws10_power'].values
        ws50 = df_power['ws50_power'].values
        ratio = ws50 / np.clip(ws10, 0.1, None)
        alpha = np.log(np.clip(ratio, 0.1, None)) / np.log(50.0 / 10.0)
        df_power['ws84_power'] = ws10 * (84.0 / 10.0) ** alpha

        temp_k = df_power['temp2m_power'].values + 273.15
        press_pa = df_power['pressure_power'].values * 100
        qv = df_power['qv2m_power'].values / 1000.0
        temp_v = temp_k * (1 + 0.61 * qv)
        df_power['air_density_power'] = press_pa / (287.05 * temp_v)

        ws84 = df_power['ws84_power'].values
        df_power['ws84_cubed_power'] = ws84 ** 3
        theory = np.zeros_like(ws84)
        reg = (ws84 >= 3.0) & (ws84 < 10.3)
        theory[reg] = 90.09 * ((ws84[reg] - 3.0) / (10.3 - 3.0)) ** 3
        rated = (ws84 >= 10.3) & (ws84 <= 25.0)
        theory[rated] = 90.09
        df_power['theory_power_total_power'] = theory

        for lag in [1, 2, 3]:
            df_power[f'ws84_lag{lag}_power'] = df_power['ws84_power'].shift(lag)
        return df_power

    train_power = process_power(train_power)
    valid_power = process_power(valid_power)

    # =============================================================================
    # 7. ОБРАБОТКА ERA5
    # =============================================================================
    def process_era5(df_era5):
        HUB_H, LOW_H, HIGH_H = 84.0, 10.0, 100.0
        ws10 = df_era5['ws10_era5'].values
        ws100 = df_era5['ws100_era5'].values
        ratio = ws100 / np.clip(ws10, 0.1, None)
        alpha = np.log(np.clip(ratio, 0.1, None)) / np.log(HIGH_H / LOW_H)
        df_era5['wind_speed_84m_era5'] = ws10 * (HUB_H / LOW_H) ** alpha

        wd10_rad = np.deg2rad(df_era5['wd10_era5'].values)
        wd100_rad = np.deg2rad(df_era5['wd100_era5'].values)
        sin10, cos10 = np.sin(wd10_rad), np.cos(wd10_rad)
        sin100, cos100 = np.sin(wd100_rad), np.cos(wd100_rad)
        w_low = (HIGH_H - HUB_H) / (HIGH_H - LOW_H)
        sin84 = sin10 * w_low + sin100 * (1 - w_low)
        cos84 = cos10 * w_low + cos100 * (1 - w_low)
        df_era5['wind_direction_84m_era5'] = np.arctan2(sin84, cos84) / (2 * np.pi) % 1.0

        df_era5['temperature_84m_era5'] = df_era5['temp2m_era5'].values - 0.65 * (84 - 2) / 100
        df_era5['pressure_msl_era5'] = df_era5['pressure_era5'].values

        ws84 = df_era5['wind_speed_84m_era5'].values
        df_era5['ws84_cubed_era5'] = ws84 ** 3
        wd84_rad = 2 * np.pi * df_era5['wind_direction_84m_era5'].values
        df_era5['u84_era5'] = ws84 * np.cos(wd84_rad)
        df_era5['v84_era5'] = ws84 * np.sin(wd84_rad)

        press = df_era5['pressure_msl_era5']
        temp84 = df_era5['temperature_84m_era5']
        df_era5['air_density_era5'] = press * 100 / (287.05 * (temp84 + 273.15))
        df_era5['wind_power_density_era5'] = 0.5 * df_era5['air_density_era5'] * df_era5['ws84_cubed_era5']

        theory = np.zeros_like(ws84)
        reg = (ws84 >= 3.0) & (ws84 < 10.3)
        theory[reg] = 90.09 * ((ws84[reg] - 3.0) / (10.3 - 3.0)) ** 3
        rated = (ws84 >= 10.3) & (ws84 <= 25.0)
        theory[rated] = 90.09
        df_era5['theory_power_total_era5'] = theory

        for lag in [1, 2, 3]:
            df_era5[f'ws84_lag{lag}_era5'] = df_era5['wind_speed_84m_era5'].shift(lag)
        wd_series = df_era5['wind_direction_84m_era5']
        for lag in [1, 2, 3]:
            wd_lag = wd_series.shift(lag)
            df_era5[f'wd84_lag{lag}_sin_era5'] = np.sin(2 * np.pi * wd_lag)
            df_era5[f'wd84_lag{lag}_cos_era5'] = np.cos(2 * np.pi * wd_lag)

        df_era5['shear_120_84_era5'] = ws100 - ws84
        df_era5['shear_84_10_era5'] = ws84 - ws10

        return df_era5

    train_era5 = process_era5(train_era5)
    valid_era5 = process_era5(valid_era5)

    # =============================================================================
    # 8. ИНТЕРПОЛЯЦИЯ ИСХОДНЫХ ДАННЫХ НА 84 м (градусы/1000 → радианы)
    # =============================================================================
    HUB_H, LOW_H, HIGH_H = 84.0, 80.0, 120.0
    for df in [train, valid]:
        ratio = df['wind_speed_120m'] / df['wind_speed_80m'].clip(lower=0.1)
        alpha = np.log(ratio.clip(lower=0.1)) / np.log(HIGH_H / LOW_H)
        df['wind_speed_84m'] = df['wind_speed_80m'] * (HUB_H / LOW_H) ** alpha

        wd80_rad = np.deg2rad(df['wind_direction_80m'] * 1000)
        wd120_rad = np.deg2rad(df['wind_direction_120m'] * 1000)
        sin_low, cos_low = np.sin(wd80_rad), np.cos(wd80_rad)
        sin_high, cos_high = np.sin(wd120_rad), np.cos(wd120_rad)
        w = (HIGH_H - HUB_H) / (HIGH_H - LOW_H)
        sin84 = sin_low * w + sin_high * (1 - w)
        cos84 = cos_low * w + cos_high * (1 - w)
        df['wind_direction_84m'] = np.arctan2(sin84, cos84) / (2 * np.pi) % 1.0

        if 'temperature_120m' in df.columns:
            df['temperature_84m'] = df['temperature_80m'] * w + df['temperature_120m'] * (1 - w)
        else:
            df['temperature_84m'] = df['temperature_80m']

    def add_base_features(df):
        df = df.copy()
        dt = df[DT_COL]
        df['dayofyear'] = dt.dt.dayofyear
        df['dayofweek'] = dt.dt.dayofweek
        df['week']      = dt.dt.isocalendar().week.astype(int)
        df['hour_sin']  = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos']  = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin']   = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['doy_cos']   = np.cos(2 * np.pi * df['dayofyear'] / 365)

        for col in ['wind_direction_10m', 'wind_direction_80m',
                    'wind_direction_120m', 'wind_direction_180m']:
            rad = np.deg2rad(df[col] * 1000)
            df[col + '_sin'] = np.sin(rad)
            df[col + '_cos'] = np.cos(rad)

        df['wind_speed_80m_sq']    = df['wind_speed_80m'] ** 2
        df['wind_speed_80m_cube']  = df['wind_speed_80m'] ** 3
        df['wind_speed_120m_cube'] = df['wind_speed_120m'] ** 3
        df['wind_shear']           = df['wind_speed_120m'] - df['wind_speed_10m']
        df['gust_ratio']           = df['wind_gusts_10m'] / (df['wind_speed_10m'] + 0.1)
        df['wind_avg']             = df[['wind_speed_80m', 'wind_speed_120m']].mean(axis=1)

        df['ws84_cubed'] = df['wind_speed_84m'] ** 3
        wd = 2 * np.pi * df['wind_direction_84m']
        df['u84'] = df['wind_speed_84m'] * np.cos(wd)
        df['v84'] = df['wind_speed_84m'] * np.sin(wd)

        if 'pressure_msl' in df.columns and 'temperature_84m' in df.columns:
            df['air_density'] = df['pressure_msl'] * 100 / (287.05 * (df['temperature_84m'] + 273.15))
            df['wind_power_density'] = 0.5 * df['air_density'] * df['ws84_cubed']

        if 'wind_speed_120m' in df.columns:
            df['shear_120_84'] = df['wind_speed_120m'] - df['wind_speed_84m']
        if 'wind_speed_10m' in df.columns:
            df['shear_84_10'] = df['wind_speed_84m'] - df['wind_speed_10m']

        ws = df['wind_speed_84m'].values
        theory = np.zeros_like(ws)
        reg = (ws >= 3.0) & (ws < 10.3)
        theory[reg] = 90.09 * ((ws[reg] - 3.0) / (10.3 - 3.0))**3
        rated = (ws >= 10.3) & (ws <= 25.0)
        theory[rated] = 90.09
        df['theory_power_total'] = theory

        for lag in [1, 2, 3]:
            df[f'ws84_lag{lag}'] = df['wind_speed_84m'].shift(lag)
        for lag in [1, 2, 3]:
            wd_lag = df['wind_direction_84m'].shift(lag)
            df[f'wd84_lag{lag}_sin'] = np.sin(2 * np.pi * wd_lag)
            df[f'wd84_lag{lag}_cos'] = np.cos(2 * np.pi * wd_lag)

        for col in ['pm2_5', 'pm10', 'co', 'aod']:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # =========================================================================
        # ИНТЕРАКТИВНЫЕ ПРИЗНАКИ РЕЛЬЕФА
        # =========================================================================
        df['elevation_median'] = median_h
        df['elevation_range'] = range_h
        df['elevation_std'] = std_h
        df['relative_elevation'] = relative_h
        df['slope_steepness'] = slope_steepness
        df['slope_aspect_sin'] = np.sin(slope_aspect)
        df['slope_aspect_cos'] = np.cos(slope_aspect)
        df['terrain_type'] = terrain_type

        if slope_steepness > 0:
            df['wind_slope_proj'] = df['u84'] * np.sin(slope_aspect) + df['v84'] * np.cos(slope_aspect)
            df['wind_slope_cross'] = -df['u84'] * np.cos(slope_aspect) + df['v84'] * np.sin(slope_aspect)
            df['wind_up_slope'] = (df['wind_slope_proj'] > 0).astype(int)
            df['wind_speed_x_slope'] = df['wind_speed_84m'] * slope_steepness
        else:
            df['wind_slope_proj'] = 0.0
            df['wind_slope_cross'] = 0.0
            df['wind_up_slope'] = 0
            df['wind_speed_x_slope'] = 0.0

        df['repair_ma6']  = df['Кол-во_ВЭУ_в_ремонте'].rolling(6, min_periods=1).mean()
        df['has_repair']  = (df['Кол-во_ВЭУ_в_ремонте'] > 0).astype(int)
        # df['repair_interp'] = df['Кол-во_ВЭУ_в_ремонте'].where(
        #     df['Кол-во_ВЭУ_в_ремонте'] != df['Кол-во_ВЭУ_в_ремонте'].shift(1)
        # ).interpolate(method='linear')
        repaired_num = df['Кол-во_ВЭУ_в_ремонте']
        df['turbines_available'] = N_TURBINES - repaired_num
        df['available_ratio']    = (N_TURBINES - repaired_num) / N_TURBINES        

        # # Определяем момент изменения значения ремонта
        # mask = df['Кол-во_ВЭУ_в_ремонте'] != df['Кол-во_ВЭУ_в_ремонте'].shift(1)
        # group_ids = mask.cumsum()
        # # df['days_since_update'] = df.groupby(group_ids).cumcount() / 24
        # hours_since = df.groupby(group_ids).cumcount()
        # df['repair_confidence'] = 1 - (hours_since / (24 * 30)).clip(lower=0.3)
        return df

    def fill_missing_180m(df):
        df = df.copy()
        df['wind_speed_180m']         = df['wind_speed_180m'].fillna(df['wind_speed_120m'])
        df['wind_direction_180m']     = df['wind_direction_180m'].fillna(df['wind_direction_120m'])
        df['wind_direction_180m_sin'] = df['wind_direction_180m_sin'].fillna(df['wind_direction_120m_sin'])
        df['wind_direction_180m_cos'] = df['wind_direction_180m_cos'].fillna(df['wind_direction_120m_cos'])
        return df

    train_f = fill_missing_180m(add_base_features(train))
    valid_f = fill_missing_180m(add_base_features(valid))

    # =============================================================================
    # 9. ОБЪЕДИНЕНИЕ ВСЕХ ИСТОЧНИКОВ
    # =============================================================================
    lag_cols = [c for c in train_f.columns if 'lag' in c and not ('_era5' in c or '_power' in c)]
    train_f = train_f.dropna(subset=lag_cols).reset_index(drop=True)
    valid_f[lag_cols] = valid_f[lag_cols].fillna(train_f[lag_cols].median())

    for src_df in [train_era5, train_power]:
        src_df[DT_COL] = src_df[DT_COL].astype(train_f[DT_COL].dtype)
        train_f = train_f.merge(src_df, on=DT_COL, how='left')

    for src_df in [valid_era5, valid_power]:
        src_df[DT_COL] = src_df[DT_COL].astype(valid_f[DT_COL].dtype)
        valid_f = valid_f.merge(src_df, on=DT_COL, how='left')

    if not train_aq.empty:
        train_aq[DT_COL] = train_aq[DT_COL].astype(train_f[DT_COL].dtype)
        train_f = train_f.merge(train_aq, on=DT_COL, how='left')
        for col in ['pm2_5', 'pm10', 'co', 'aod']:
            if col in train_f.columns:
                train_f[col] = train_f[col].fillna(train_f[col].median())
            else:
                train_f[col] = 0.0
    else:
        for col in ['pm2_5', 'pm10', 'co', 'aod']:
            train_f[col] = 0.0

    if not valid_aq.empty:
        valid_aq[DT_COL] = valid_aq[DT_COL].astype(valid_f[DT_COL].dtype)
        valid_f = valid_f.merge(valid_aq, on=DT_COL, how='left')
        for col in ['pm2_5', 'pm10', 'co', 'aod']:
            if col in valid_f.columns:
                valid_f[col] = valid_f[col].fillna(train_f[col].median() if col in train_f.columns else 0.0)
            else:
                valid_f[col] = 0.0
    else:
        for col in ['pm2_5', 'pm10', 'co', 'aod']:
            valid_f[col] = 0.0

    for suffix in ['_era5', '_power']:
        cols = [c for c in train_f.columns if c.endswith(suffix)]
        train_f[cols] = train_f[cols].fillna(train_f[cols].median())
        valid_f[cols] = valid_f[cols].fillna(train_f[cols].median())

    # =============================================================================
    # 10. WAKE EFFECT – 72 сектора
    # =============================================================================
    dir_84 = train_f['wind_direction_84m']
    sector = (dir_84 * N_WAKE_SECTORS).astype(int) % N_WAKE_SECTORS
    fact_vs_theory = train_f[TARGET] / (train_f['theory_power_total'] + 1e-6)
    wake_factors = fact_vs_theory.groupby(sector).mean()

    train_f['wake_factor'] = (train_f['wind_direction_84m'] * N_WAKE_SECTORS).astype(int) % N_WAKE_SECTORS
    train_f['wake_factor'] = train_f['wake_factor'].map(wake_factors)
    valid_f['wake_factor'] = (valid_f['wind_direction_84m'] * N_WAKE_SECTORS).astype(int) % N_WAKE_SECTORS
    valid_f['wake_factor'] = valid_f['wake_factor'].map(wake_factors)

    train_f['wake_corrected_power'] = train_f['theory_power_total'] * train_f['wake_factor']
    valid_f['wake_corrected_power'] = valid_f['theory_power_total'] * valid_f['wake_factor']

    # =============================================================================
    # 11. ФОРМИРУЕМ МАССИВЫ
    # =============================================================================
    EXCLUDE_FEATURES = [TARGET, DT_COL, 'Кол-во_ВЭУ_в_ремонте']
    FEATURES = [c for c in train_f.columns if c not in EXCLUDE_FEATURES]
    print(f"Финальное количество признаков: {len(FEATURES)}")

    X = train_f[FEATURES].values
    y = train_f[TARGET].values
    X_valid = valid_f[FEATURES].values
    





    
    
    
    # # Оптимизация LightGBM (только если модель активна)
    # if MODEL_CONFIGS['LightGBM']['seeds']:
    #     # best_lgb_params = optimize_lightgbm(X, y)
    #     best_lgb_params = {'n_estimators': 1453, 'learning_rate': 0.013840925580516236, 'num_leaves': 86, 'max_depth': 6, 'min_child_samples': 43, 'subsample': 0.768182043222019, 'colsample_bytree': 0.736109480670197, 'reg_alpha': 0.0010719709736106663, 'reg_lambda': 0.9126523093098456, 'objective': 'quantile', 'alpha': 0.5, 'n_jobs': -1, 'verbose': -1}
    #     # Обновляем параметры LightGBM
    #     MODEL_CONFIGS['LightGBM']['params'].update(best_lgb_params)
    #     # Пересоздаем обертку с новыми параметрами
    #     model_wrappers['LightGBM'] = LightGBMWrapper('LightGBM', MODEL_CONFIGS['LightGBM'])

    # =============================================================================
    # ОПЦИОНАЛЬНАЯ ОПТИМИЗАЦИЯ (раскомментировать для поиска параметров)
    # =============================================================================
    # ВНИМАНИЕ: оптимизация занимает время. Найденные параметры нужно скопировать
    # в конфиг MODEL_CONFIGS, после чего эту секцию можно закомментировать.

    best_lgb_params = None
    best_cat_params = None
    best_xgb_params = None

    # Лучшие найденные параметры (вставьте сюда после оптимизации):
    # best_lgb_params = {'n_estimators': 1453, 'learning_rate': 0.013840925580516236, 'num_leaves': 86, 'max_depth': 6, 'min_child_samples': 43, 'subsample': 0.768182043222019, 'colsample_bytree': 0.736109480670197, 'reg_alpha': 0.0010719709736106663, 'reg_lambda': 0.9126523093098456, 'objective': 'quantile', 'alpha': 0.5, 'n_jobs': -1, 'verbose': -1}
    # best_cat_params = {'iterations': 1362, 'learning_rate': 0.013597596978785926, 'depth': 7, 'l2_leaf_reg': 0.02114815090518835, 'border_count': 101, 'bootstrap_type': 'Bernoulli', 'subsample': 0.9035862570363855, 'verbose': False}
    best_xgb_params = {'n_estimators': 1071, 'learning_rate': 0.026569216347370212, 'max_depth': 5, 'min_child_weight': 10, 'subsample': 0.8097813488756438, 'colsample_bytree': 0.6463475564516007, 'colsample_bylevel': 0.8496268024138052, 'reg_alpha': 0.0251343211987505, 'reg_lambda': 0.03247801858119664, 'gamma': 0.009378112717555705, 'objective': 'reg:quantileerror', 'quantile_alpha': 0.5, 'verbosity': 0}

    best_lgb_params = {'n_estimators': 1309, 'learning_rate': 0.011548797797512488, 'num_leaves': 42, 'max_depth': 11, 'min_child_samples': 37, 'subsample': 0.6805130457867344, 'colsample_bytree': 0.7114480131980546, 'reg_alpha': 0.008808333526206925, 'reg_lambda': 0.010302057297292477, 'objective': 'quantile', 'alpha': 0.5, 'n_jobs': -1, 'verbose': -1}

    # РАСКОММЕНТИРОВАТЬ ДЛЯ ОПТИМИЗАЦИИ:
    # best_lgb_params = optimize_lightgbm(X, y)
    # best_cat_params = optimize_catboost(X, y)
    # best_xgb_params = optimize_xgboost(X, y)


    # Применяем найденные параметры к конфигам
    if best_lgb_params is not None:
        MODEL_CONFIGS['LightGBM']['params'].update(best_lgb_params)
        print("✅ Применены оптимизированные параметры LightGBM")

    if best_cat_params is not None:
        MODEL_CONFIGS['CatBoost']['params'].update(best_cat_params)
        print("✅ Применены оптимизированные параметры CatBoost")

    if best_xgb_params is not None:
        MODEL_CONFIGS['XGBoost']['params'].update(best_xgb_params)
        print("✅ Применены оптимизированные параметры XGBoost")


    # Создаем обертки для моделей
    model_wrappers = create_model_wrappers()
    
    # Создаем holdout для подбора весов
    holdout_mask = (train_f[DT_COL] >= '2025-01-01') & (train_f[DT_COL] < '2025-04-01')
    X_tr, X_ho = X[~holdout_mask], X[holdout_mask]
    y_tr, y_ho = y[~holdout_mask], y[holdout_mask]
    
    # Обучаем модели на train с валидацией и подбираем веса
    best_weights, model_wrappers = train_and_optimize_weights(X_tr, y_tr, X_ho, y_ho, model_wrappers)
    
    # Финальное обучение только моделей с весом > 0
    final_pred = final_train_and_predict(X, y, X_valid, model_wrappers, best_weights, optimize_lgbm_iters=False)
    
    # Постобработка прогноза (оригинальная логика)
    zero_mask = valid_f['turbines_available'].values <= 0
    final_pred[zero_mask] = 0.0
    final_pred = np.minimum(final_pred, CAPACITY * valid_f['available_ratio'].values + 1e-3)
    final_pred = np.clip(final_pred, 0, CAPACITY)
    
    # Сохранение результатов
    result = (pd.DataFrame({DT_COL: valid_f[DT_COL].values, TARGET: final_pred})
                .set_index(DT_COL).loc[valid_order].reset_index())
    result[[TARGET]].to_csv(OUTPUT_VALID, index=False)
    print(f"\n[OK] {OUTPUT_VALID} сохранён ({len(result)} строк)")

if __name__ == "__main__":
    main()