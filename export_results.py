import os, json
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

# ---- optional XGBoost ----
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ====== 你必須對齊你的 CSV 欄位名稱 ======
FEATURE_COLS = [
    "Age","EducationLevel","EmploymentStatus","JobTitle","SalaryCategory","AnnualIncome",
    "NetWorth","CreditScore","RiskRating","AccountBalance","NumBankProducts",
    "HasCreditCard","HasMortgage","HasPersonalLoan","HasLifeInsurance","HasMutualFunds",
    "InvestmentPortfolioValue"
]

BINARY_COLS = ["HasCreditCard","HasMortgage","HasPersonalLoan","HasLifeInsurance","HasMutualFunds"]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Yes/No -> 1/0
    yn_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].map(yn_map).fillna(df[c])

    # 2) 把數值欄位強制轉成 numeric（轉不了的變 NaN）
    for c in FEATURE_COLS:
        if c in df.columns and c not in CAT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) 把 target 也轉 numeric（保險）
    for t in TARGET_COLS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")

    # 4) 丟掉含 NaN 的列（最簡單粗暴且穩）
    keep_cols = FEATURE_COLS + TARGET_COLS
    before = len(df)
    df = df.dropna(subset=keep_cols).reset_index(drop=True)
    after = len(df)
    print(f"[clean_df] dropped rows with NaN: {before-after} (remain {after})")

    return df



TARGET_COLS = ['US_Equity', 'International_Equity', 'Bonds', 'REIT', 'Cash']

CAT_COLS = ["EducationLevel","EmploymentStatus","JobTitle","SalaryCategory","RiskRating"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

DATA_PATH = r"C:\Users\user\Desktop\ML\project\Bank_Marketing_Split_dataset_with_allocations.csv"   # <-- 改成你的檔名
OUT_DIR = "results"
TEST_SIZE = 0.2
SEED = 42


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))),
        "mae":  float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average")),
        "r2":   float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }

def per_target_metrics(y_true, y_pred):
    out = []
    for i, t in enumerate(TARGET_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out.append({
            "target": t,
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae":  float(mean_absolute_error(yt, yp)),
            "r2":   float(r2_score(yt, yp)),
        })
    return out

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop"
    )

def get_feature_names(pre):
    names = []
    names.extend(NUM_COLS)
    ohe = pre.named_transformers_["cat"]
    names.extend(list(ohe.get_feature_names_out(CAT_COLS)))
    return names

def importance_from_pipe(pipe, model_kind: str):
    """
    model_kind: 'linear' | 'rf' | 'xgb'
    Return top-30 feature importances as list of dict.
    """
    pre = pipe.named_steps["preprocess"]
    mo  = pipe.named_steps["model"]  # MultiOutputRegressor
    feat_names = get_feature_names(pre)

    if model_kind == "linear":
        vals = [np.abs(est.coef_) for est in mo.estimators_]
        imp = np.mean(np.vstack(vals), axis=0)
    else:
        vals = [est.feature_importances_ for est in mo.estimators_]
        imp = np.mean(np.vstack(vals), axis=0)

    df_imp = pd.DataFrame({"feature": feat_names, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False).head(30)
    return df_imp.to_dict("records")

def run_and_export(model_name: str, base_estimator, model_kind: str, df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", MultiOutputRegressor(base_estimator)),
    ])

    pipe.fit(X_tr, y_tr)

    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)
    pred_all = pipe.predict(X)

    result = {
        "model_name": model_name,
        "n_rows": int(len(df)),
        "metrics": {
            "train": metrics(y_tr.values, pred_tr),
            "test":  metrics(y_te.values, pred_te),
        },
        "per_target": {
            "train": per_target_metrics(y_tr.values, pred_tr),
            "test":  per_target_metrics(y_te.values, pred_te),
        },
        "importance": importance_from_pipe(pipe, model_kind),
        "pred_all": pred_all.tolist(),
        "y_true_all": y.values.tolist(),
        # 你若想讓 dashboard 顯示某 row 的原始 features，可一併存（可選）
        # "X_all": X.to_dict("records")
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"✅ saved {out_path}")


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_df(df)

    # 基本欄位檢查
    miss_f = [c for c in FEATURE_COLS if c not in df.columns]
    miss_t = [c for c in TARGET_COLS if c not in df.columns]
    if miss_f or miss_t:
        raise ValueError(f"Missing columns\nfeatures={miss_f}\ntargets={miss_t}")

    # --- LinearRegression ---
    run_and_export(
        "LinearRegression",
        LinearRegression(),
        model_kind="linear",
        df=df
    )

    # --- RandomForest ---
    run_and_export(
        "RandomForest",
        RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        model_kind="rf",
        df=df
    )

    # --- XGBoost ---
    if HAS_XGB:
        run_and_export(
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
                objective="reg:squarederror",
            ),
            model_kind="xgb",
            df=df
        )
    else:
        print("⚠️ xgboost not installed, skip XGBoost.json (pip install xgboost)")

if __name__ == "__main__":
    main()
