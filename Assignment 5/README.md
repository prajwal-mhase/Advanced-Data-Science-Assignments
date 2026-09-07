# Assignment 5: Online Shopper Revenue Deployment

This practical deploys a machine-learning model trained on `online_shoppers_intention.csv`.
The model predicts whether an online shopping session will generate revenue.

## 1. Install packages

```powershell
py -m pip install -r requirements.txt
```

## 2. Run the Flask web application

```powershell
py app.py
```

`app.py` trains and saves `shopper_revenue_model.joblib` automatically if the model file does not already exist. The saved pipeline contains both preprocessing and the logistic-regression classifier.

Open `http://127.0.0.1:5000/` in a browser. The light-theme page provides a small form for the most important shopping-session signals.

The REST endpoint is also available at `POST http://127.0.0.1:5000/predict`. Send JSON containing the 17 input features. Example:

```json
{
  "Administrative": 0,
  "Administrative_Duration": 0,
  "Informational": 0,
  "Informational_Duration": 0,
  "ProductRelated": 10,
  "ProductRelated_Duration": 627.5,
  "BounceRates": 0.02,
  "ExitRates": 0.05,
  "PageValues": 0,
  "SpecialDay": 0,
  "Month": "Feb",
  "OperatingSystems": 3,
  "Browser": 3,
  "Region": 1,
  "TrafficType": 4,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

The response contains the predicted revenue status and probability. The browser form uses the same model through the `/web-predict` route.

## Conclusion

The model was serialized with joblib and deployed through a Flask web application and `/predict` REST endpoint. Both routes load the same trained pipeline, so the training preprocessing is reused during prediction.
