from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FILE = Path(__file__).with_name("shopper_revenue_model.joblib")
DATA_FILE = Path(__file__).with_name("online_shoppers_intention.csv")
FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]

app = Flask(__name__)


def load_model():
        if MODEL_FILE.exists():
                return joblib.load(MODEL_FILE)

        data = pd.read_csv(DATA_FILE)
        numeric_features = [feature for feature in FEATURES if feature not in ["Month", "VisitorType"]]
        preprocessor = ColumnTransformer([
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Month", "VisitorType"]),
        ])
        pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
        ])
        pipeline.fit(data[FEATURES], data["Revenue"])
        joblib.dump(pipeline, MODEL_FILE)
        return pipeline


model = load_model()

PAGE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Shopper Signal</title>
    <style>
        :root { --ink:#17212b; --muted:#66717d; --blue:#1769e0; --dark:#0b4db4; --line:#dce4ec; --green:#16834b; }
        * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#eef4f8; font:15px/1.5 Arial,sans-serif; }
        .shell { max-width:1080px; margin:auto; padding:40px 20px 56px; } .hero { display:flex; justify-content:space-between; gap:28px; align-items:end; margin-bottom:28px; }
        .eyebrow { color:var(--blue); font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; } h1 { margin:8px 0; max-width:600px; font-size:clamp(30px,5vw,52px); line-height:1.03; letter-spacing:-1px; }
        .intro { max-width:580px; margin:0; color:var(--muted); font-size:17px; } .badge { padding:10px 14px; border:1px solid #cfe0ef; border-radius:999px; color:var(--dark); background:#fff; font-weight:700; white-space:nowrap; }
        .layout { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(260px,.6fr); gap:18px; } .panel { border:1px solid var(--line); border-radius:16px; background:#fff; box-shadow:0 12px 36px #26466114; }
        .form-panel,.result { padding:26px; } .panel h2 { margin:0 0 5px; font-size:20px; } .panel-copy { margin:0 0 22px; color:var(--muted); }
        .fields { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; } label { display:block; color:#3e4b57; font-size:13px; font-weight:700; }
        input,select { width:100%; margin-top:7px; padding:11px 12px; border:1px solid var(--line); border-radius:8px; color:var(--ink); background:#fbfdff; font:inherit; } input:focus,select:focus { outline:3px solid #dceaff; border-color:var(--blue); }
        .wide { grid-column:1/-1; } .check { display:flex; gap:10px; align-items:center; } .check input { width:17px; height:17px; margin:0; accent-color:var(--blue); }
        button { width:100%; margin-top:22px; padding:13px 18px; border:0; border-radius:8px; color:#fff; background:var(--blue); font:inherit; font-weight:700; cursor:pointer; } button:hover { background:var(--dark); }
        .result { display:flex; min-height:100%; flex-direction:column; justify-content:space-between; background:#f8fbff; } .result h2 { color:#456070; }
        .result-value { margin:26px 0 8px; color:var(--green); font-size:32px; font-weight:800; line-height:1.1; } .neutral { color:#bd6c17; } .probability { color:var(--muted); }
        .note { padding-top:24px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; } @media(max-width:720px) { .hero { display:block; } .badge { display:inline-block; margin-top:18px; } .layout { grid-template-columns:1fr; } }
    </style>
</head>
<body><main class="shell"><header class="hero"><div><div class="eyebrow">Flask model deployment</div><h1>Understand a shopper session at a glance.</h1><p class="intro">Enter a few session signals and estimate whether the visit is likely to generate revenue.</p></div><div class="badge">Online Shopper Signal</div></header>
<section class="layout"><form class="panel form-panel" method="post" action="/web-predict"><h2>Session details</h2><p class="panel-copy">Use the main browsing signals for a quick prediction.</p><div class="fields">
    <label>Product-related pages<input name="ProductRelated" type="number" min="0" value="{{ form.get('ProductRelated',10) if form else 10 }}" required></label><label>Product browsing time<input name="ProductRelated_Duration" type="number" min="0" step="0.1" value="{{ form.get('ProductRelated_Duration',120) if form else 120 }}" required></label>
    <label>Page value<input name="PageValues" type="number" min="0" step="0.01" value="{{ form.get('PageValues',0) if form else 0 }}" required></label><label>Exit rate<input name="ExitRates" type="number" min="0" max="1" step="0.001" value="{{ form.get('ExitRates',0.05) if form else 0.05 }}" required></label>
    <label>Month<select name="Month"><option>Feb</option><option>Mar</option><option>May</option><option>June</option><option>Jul</option><option>Aug</option><option>Sep</option><option>Oct</option><option>Nov</option><option>Dec</option></select></label><label>Visitor type<select name="VisitorType"><option>Returning_Visitor</option><option>New_Visitor</option><option>Other</option></select></label>
    <input type="hidden" name="Administrative" value="0"><input type="hidden" name="Administrative_Duration" value="0"><input type="hidden" name="Informational" value="0"><input type="hidden" name="Informational_Duration" value="0"><input type="hidden" name="BounceRates" value="0.05"><input type="hidden" name="SpecialDay" value="0"><input type="hidden" name="OperatingSystems" value="2"><input type="hidden" name="Browser" value="2"><input type="hidden" name="Region" value="1"><input type="hidden" name="TrafficType" value="2"><label class="check wide"><input name="Weekend" type="checkbox"> Weekend visit</label>
</div><button type="submit">Predict revenue</button></form><aside class="panel result"><div><h2>Prediction</h2>{% if result %}<div class="result-value {% if not result.revenue %}neutral{% endif %}">{{ result.prediction }}</div><div class="probability">Revenue probability: <strong>{{ "%.1f"|format(result.probability * 100) }}%</strong></div>{% else %}<div class="result-value neutral">Waiting for input</div><div class="probability">Your result will appear here.</div>{% endif %}</div><p class="note">A learning demonstration powered by the trained logistic-regression model.</p></aside></section></main></body></html>"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.post("/web-predict")
def web_predict():
    data = request.form.to_dict()
    data["Weekend"] = "Weekend" in request.form
    data["SpecialDay"] = float(data.get("SpecialDay", 0))
    for feature in [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "OperatingSystems",
        "Browser", "Region", "TrafficType",
    ]:
        data[feature] = float(data[feature])

    result = predict_prediction(data)
    return render_template_string(PAGE, result=result, form=data)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({"error": "Missing features", "features": missing}), 400

    try:
        return jsonify(predict_prediction(data))
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


def predict_prediction(data):
    sample = pd.DataFrame([[data[feature] for feature in FEATURES]], columns=FEATURES)
    prediction = bool(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    return {
        "prediction": "Revenue likely" if prediction else "Revenue unlikely",
        "revenue": prediction,
        "probability": round(probability, 4),
    }



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)