# model_runners.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# === 請對齊你資料集實際欄位 ===
FEATURE_COLS = [
    "Age","EducationLevel","EmploymentStatus","JobTitle","SalaryCategory","AnnualIncome",
    "NetWorth","CreditScore","RiskRating","AccountBalance","NumBankProducts",
    "HasCreditCard","HasMortgage","HasPersonalLoan","HasLifeInsurance","HasMutualFunds",
    "InvestmentPortfolioValue"
]
TARGET_COLS = ["US_Equity","Intl_Equity","Bonds","REIT","Cash"]

CAT_COLS = ["EducationLevel","EmploymentStatus","JobTitle","SalaryCategory","RiskRating"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop"
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))
    mae  = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average"))
    r2   = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> List[Dict[str, Any]]:
    out = []
    for i, t in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out.append({
            "target": t,
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae":  float(mean_absolute_error(yt, yp)),
            "r2":   float(r2_score(yt, yp)),
        })
    return out


def _feature_names(pre: ColumnTransformer) -> List[str]:
    # num names
    names = []
    names.extend(NUM_COLS)
    # cat onehot names
    ohe = pre.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(CAT_COLS))
    names.extend(ohe_names)
    return names


def _importance_from_multioutput(pipe: Pipeline, model_type: str) -> pd.DataFrame:
    pre = pipe.named_steps["preprocess"]
    mo  = pipe.named_steps["model"]  # MultiOutputRegressor
    fn  = _feature_names(pre)

    if model_type == "linear":
        vals = []
        for est in mo.estimators_:
            vals.append(np.abs(est.coef_))
        imp = np.mean(np.vstack(vals), axis=0)

    elif model_type in ("rf", "xgb"):
        vals = []
        for est in mo.estimators_:
            vals.append(est.feature_importances_)
        imp = np.mean(np.vstack(vals), axis=0)

    else:
        imp = np.zeros(len(fn), dtype=float)

    df = pd.DataFrame({"feature": fn, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _train_eval_common(
    df: pd.DataFrame,
    model_name: str,
    base_estimator,
    model_type: str,
    test_size: float = 0.2,
    seed: int = 42
) -> Dict[str, Any]:

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    pipe = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("model", MultiOutputRegressor(base_estimator)),
    ])

    pipe.fit(X_tr, y_tr)

    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)

    m_tr = _metrics(y_tr.values, pred_tr)
    m_te = _metrics(y_te.values, pred_te)

    per_t_tr = _per_target_metrics(y_tr.values, pred_tr, TARGET_COLS)
    per_t_te = _per_target_metrics(y_te.values, pred_te, TARGET_COLS)

    # 用全資料做預測（dashboard 可用來挑 row 比較）
    pred_all = pipe.predict(X)

    imp_df = _importance_from_multioutput(pipe, model_type).head(30)

    return {
        "model_name": model_name,
        "metrics": {"train": m_tr, "test": m_te},
        "per_target": {"train": per_t_tr, "test": per_t_te},
        "pred_all": pred_all.tolist(),
        "y_true_all": y.values.tolist(),
        "importance": imp_df.to_dict("records"),
        "n_rows": int(len(df)),
    }


# ========= 你要在 dashboard 顯示的模型們：把你 code files 的訓練邏輯放進來 =========

def run_linear(df: pd.DataFrame, test_size=0.2, seed=42) -> Dict[str, Any]:
    # ✅ 這裡等價於你 notebook 的 Linear Regression pipeline
    est = LinearRegression()
    return _train_eval_common(df, "LinearRegression", est, model_type="linear", test_size=test_size, seed=seed)


def run_random_forest(df: pd.DataFrame, test_size=0.2, seed=42) -> Dict[str, Any]:
    # ✅ 這裡等價於你 notebook 的 RandomForest
    est = RandomForestRegressor(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1
    )
    return _train_eval_common(df, "RandomForest", est, model_type="rf", test_size=test_size, seed=seed)


def run_xgboost(df: pd.DataFrame, test_size=0.2, seed=42) -> Dict[str, Any]:
    if not HAS_XGB:
        raise RuntimeError("xgboost 未安裝：pip install xgboost")
    # ✅ 這裡等價於你 notebook 的 XGBoost（可改成你實際參數）
    est = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    return _train_eval_common(df, "XGBoost", est, model_type="xgb", test_size=test_size, seed=seed)


def available_models() -> Dict[str, Any]:
    models = {
        "LinearRegression": run_linear,
        "RandomForest": run_random_forest,
    }
    if HAS_XGB:
        models["XGBoost"] = run_xgboost
    return models
