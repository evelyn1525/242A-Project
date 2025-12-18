# # app.py
# import base64, io
# import pandas as pd

# from dash import Dash, dcc, html, Input, Output, State
# import plotly.express as px
# import plotly.graph_objects as go

# from model_runners import (
#     FEATURE_COLS, TARGET_COLS,
#     available_models
# )

# app = Dash(__name__)
# server = app.server  # 部署用 gunicorn app:server

# def decode_upload(contents: str) -> pd.DataFrame:
#     _, content_string = contents.split(",")
#     decoded = base64.b64decode(content_string)
#     try:
#         return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
#     except UnicodeDecodeError:
#         return pd.read_csv(io.StringIO(decoded.decode("latin-1")))

# def validate_df(df: pd.DataFrame):
#     miss_f = [c for c in FEATURE_COLS if c not in df.columns]
#     miss_t = [c for c in TARGET_COLS if c not in df.columns]
#     if miss_f or miss_t:
#         raise ValueError(f"Missing columns\nfeatures={miss_f}\ntargets={miss_t}")

# app.layout = html.Div([
#     html.H2("Model Comparison Dashboard (Dash/Plotly)"),

#     dcc.Upload(
#         id="upload",
#         children=html.Div(["拖拉或點擊上傳 CSV（需含 features + targets）"]),
#         style={"width":"100%","height":"70px","lineHeight":"70px",
#                "borderWidth":"2px","borderStyle":"dashed","borderRadius":"10px",
#                "textAlign":"center"},
#         multiple=False
#     ),
#     html.Div(id="status", style={"whiteSpace":"pre-wrap", "marginTop":"8px"}),

#     html.Div([
#         html.Div([
#             html.Label("Test size"),
#             dcc.Slider(id="test_size", min=0.1, max=0.4, step=0.05, value=0.2,
#                        marks={0.1:"0.1",0.2:"0.2",0.3:"0.3",0.4:"0.4"})
#         ], style={"width":"48%", "display":"inline-block"}),

#         html.Div([
#             html.Label("Seed"),
#             dcc.Input(id="seed", type="number", value=42, style={"width":"100%"})
#         ], style={"width":"48%", "display":"inline-block", "paddingLeft":"18px"})
#     ], style={"marginTop":"12px"}),

#     html.Button("Run all models", id="run_btn", n_clicks=0, style={"marginTop":"12px"}),

#     dcc.Store(id="df_store"),
#     dcc.Store(id="results_store"),

#     html.Hr(),

#     dcc.Tabs([
#         dcc.Tab(label="Overview", children=[
#             html.H4("Test Metrics Summary"),
#             dcc.Graph(id="metrics_bar"),
#             html.H4("Per-target (Test)"),
#             dcc.Dropdown(id="metric_pick", options=[
#                 {"label":"RMSE", "value":"rmse"},
#                 {"label":"MAE", "value":"mae"},
#                 {"label":"R²", "value":"r2"},
#             ], value="rmse", clearable=False),
#             dcc.Graph(id="per_target_heat"),
#         ]),
#         dcc.Tab(label="Predictions Explorer", children=[
#             html.Div([
#                 html.Div([
#                     html.Label("Select model"),
#                     dcc.Dropdown(id="model_select", clearable=False)
#                 ], style={"width":"48%","display":"inline-block"}),
#                 html.Div([
#                     html.Label("Row index"),
#                     dcc.Slider(id="row_idx", min=0, max=0, step=1, value=0)
#                 ], style={"width":"48%","display":"inline-block","paddingLeft":"18px"})
#             ], style={"marginTop":"10px"}),
#             dcc.Graph(id="row_compare"),
#         ]),
#         dcc.Tab(label="Feature Importance", children=[
#             html.Div([
#                 html.Label("Select model"),
#                 dcc.Dropdown(id="model_select_imp", clearable=False),
#             ], style={"width":"50%"}),
#             dcc.Graph(id="imp_plot")
#         ])
#     ])
# ], style={"maxWidth":"1200px","margin":"0 auto","padding":"18px"})


# @app.callback(
#     Output("df_store", "data"),
#     Output("status", "children"),
#     Input("upload", "contents"),
#     State("upload", "filename"),
# )
# def on_upload(contents, filename):
#     if not contents:
#         return None, "尚未上傳。"
#     try:
#         df = decode_upload(contents)
#         validate_df(df)
#         return df.to_json(orient="split"), f"✅ 已載入 {filename}\nrows={len(df)}, cols={len(df.columns)}"
#     except Exception as e:
#         return None, f"❌ 上傳/解析失敗：{e}"


# @app.callback(
#     Output("results_store", "data"),
#     Input("run_btn", "n_clicks"),
#     State("df_store", "data"),
#     State("test_size", "value"),
#     State("seed", "value"),
# )
# def run_all_models(n, df_json, test_size, seed):
#     if not df_json:
#         return None
#     df = pd.read_json(df_json, orient="split")

#     results = {}
#     models = available_models()
#     for name, fn in models.items():
#         results[name] = fn(df, test_size=float(test_size), seed=int(seed))
#     return results


# @app.callback(
#     Output("metrics_bar", "figure"),
#     Output("per_target_heat", "figure"),
#     Output("model_select", "options"),
#     Output("model_select", "value"),
#     Output("model_select_imp", "options"),
#     Output("model_select_imp", "value"),
#     Output("row_idx", "max"),
#     Output("row_idx", "value"),
#     Input("results_store", "data"),
#     State("metric_pick", "value"),
# )
# def render_overview(res, metric_pick):
#     if not res:
#         return go.Figure(), go.Figure(), [], None, [], None, 0, 0

#     model_names = list(res.keys())

#     # --- Metrics bar (test) ---
#     dfm = pd.DataFrame([
#         {"model": m,
#          "rmse": res[m]["metrics"]["test"]["rmse"],
#          "mae":  res[m]["metrics"]["test"]["mae"],
#          "r2":   res[m]["metrics"]["test"]["r2"]}
#         for m in model_names
#     ])
#     fig_bar = px.bar(dfm, x="model", y=["rmse","mae","r2"], barmode="group",
#                      title="Test Metrics (lower RMSE/MAE better, higher R² better)")

#     # --- Per-target heatmap ---
#     rows = []
#     for m in model_names:
#         for trow in res[m]["per_target"]["test"]:
#             rows.append({"model": m, "target": trow["target"], metric_pick: trow[metric_pick]})
#     dft = pd.DataFrame(rows).pivot(index="model", columns="target", values=metric_pick)
#     fig_heat = px.imshow(dft, text_auto=".3f", aspect="auto",
#                          title=f"Per-target Test {metric_pick.upper()}")

#     opts = [{"label": m, "value": m} for m in model_names]
#     default_model = model_names[0]
#     n_rows = res[default_model]["n_rows"]
#     return fig_bar, fig_heat, opts, default_model, opts, default_model, max(0, n_rows-1), 0


# @app.callback(
#     Output("row_compare", "figure"),
#     Input("model_select", "value"),
#     Input("row_idx", "value"),
#     State("results_store", "data"),
# )
# def render_row_compare(model_name, idx, res):
#     if not res or not model_name:
#         return go.Figure()
#     idx = int(idx or 0)

#     y_true = res[model_name]["y_true_all"][idx]
#     y_pred = res[model_name]["pred_all"][idx]

#     fig = go.Figure()
#     fig.add_trace(go.Bar(name="True", x=TARGET_COLS, y=y_true))
#     fig.add_trace(go.Bar(name="Pred", x=TARGET_COLS, y=y_pred))
#     fig.update_layout(barmode="group", title=f"{model_name} | Row {idx}: True vs Pred Allocation")
#     return fig


# @app.callback(
#     Output("imp_plot", "figure"),
#     Input("model_select_imp", "value"),
#     State("results_store", "data"),
# )
# def render_importance(model_name, res):
#     if not res or not model_name:
#         return go.Figure()
#     imp = pd.DataFrame(res[model_name]["importance"])
#     if imp.empty:
#         return go.Figure()
#     fig = px.bar(imp[::-1], x="importance", y="feature", orientation="h",
#                  title=f"{model_name} | Top Feature Importance")
#     return fig


# if __name__ == "__main__":
#     # Windows 本機跑：用 127.0.0.1（不要用 0.0.0.0）
#     app.run(host="127.0.0.1", port=8050, debug=True)


# app.py (read-only dashboard)
import os, json
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = "results"
TARGET_COLS = ["US_Equity","Intl_Equity","Bonds","REIT","Cash"]

def list_models():
    if not os.path.exists(RESULTS_DIR):
        return []
    return sorted([f[:-5] for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])

def load_result(model_name: str):
    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Read-only Model Results Dashboard"),

    html.Div([
        html.Label("Select model result file"),
        dcc.Dropdown(id="model_dd", options=[{"label":m, "value":m} for m in list_models()],
                     value=(list_models()[0] if list_models() else None),
                     clearable=False),
    ], style={"width":"50%"}),

    html.Hr(),

    html.Div([
        html.H4("Metrics (Train/Test)"),
        html.Pre(id="metrics_box", style={"whiteSpace":"pre-wrap"})
    ]),

    html.Hr(),

    html.Div([
        html.Div([
            html.H4("Per-target metric (Test)"),
            dcc.Dropdown(id="metric_dd",
                         options=[{"label":"RMSE","value":"rmse"},{"label":"MAE","value":"mae"},{"label":"R²","value":"r2"}],
                         value="rmse", clearable=False),
            dcc.Graph(id="per_target_plot")
        ], style={"width":"55%","display":"inline-block"}),

        html.Div([
            html.H4("Feature Importance (Top 25)"),
            dcc.Graph(id="imp_plot")
        ], style={"width":"43%","display":"inline-block","paddingLeft":"18px"})
    ]),

    html.Hr(),

    html.Div([
        html.H4("Row inspection (True vs Pred)"),
        dcc.Slider(id="row_idx", min=0, max=0, step=1, value=0),
        dcc.Graph(id="row_compare")
    ])
], style={"maxWidth":"1200px","margin":"0 auto","padding":"18px"})


@app.callback(
    Output("metrics_box", "children"),
    Output("per_target_plot", "figure"),
    Output("imp_plot", "figure"),
    Output("row_idx", "max"),
    Output("row_idx", "value"),
    Input("model_dd", "value"),
    Input("metric_dd", "value"),
)
def render(model_name, metric_name):
    if not model_name:
        return "No result files found in ./results", go.Figure(), go.Figure(), 0, 0

    res = load_result(model_name)

    m = res["metrics"]
    txt = (
        f"Model: {res['model_name']}\n\n"
        f"[Train] RMSE={m['train']['rmse']:.4f}  MAE={m['train']['mae']:.4f}  R²={m['train']['r2']:.4f}\n"
        f"[Test ] RMSE={m['test']['rmse']:.4f}  MAE={m['test']['mae']:.4f}  R²={m['test']['r2']:.4f}\n"
    )

    # per-target bar (test)
    dft = pd.DataFrame(res["per_target"]["test"])
    fig_t = px.bar(dft, x="target", y=metric_name, title=f"Per-target Test {metric_name.upper()} ({model_name})")

    # importance
    imp = pd.DataFrame(res.get("importance", []))
    if not imp.empty:
        imp = imp.head(25)
        fig_imp = px.bar(imp[::-1], x="importance", y="feature", orientation="h",
                         title=f"Top-25 Importance ({model_name})")
    else:
        fig_imp = go.Figure()

    n = res.get("n_rows", len(res.get("pred_all", [])))
    return txt, fig_t, fig_imp, max(0, n-1), 0


@app.callback(
    Output("row_compare", "figure"),
    Input("model_dd", "value"),
    Input("row_idx", "value"),
)
def row_plot(model_name, idx):
    if not model_name:
        return go.Figure()
    res = load_result(model_name)
    idx = int(idx or 0)

    y_true = res["y_true_all"][idx]
    y_pred = res["pred_all"][idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="True", x=TARGET_COLS, y=y_true))
    fig.add_trace(go.Bar(name="Pred", x=TARGET_COLS, y=y_pred))
    fig.update_layout(barmode="group", title=f"{model_name} | Row {idx}: True vs Pred")
    return fig


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
