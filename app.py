import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
# Specific import for loaded ARIMA model to ensure correct method access
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
import logging

# Set up basic logging for Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Global variables for models and feature lists --- 
loaded_department_model = None
loaded_priority_threshold = None
loaded_high_priority_model = None # Used for High Priority Prediction (section I) AND Auto-Tagging Priority (section III)
loaded_incident_volume_model = None
loaded_rfc_rule_model = None
loaded_failure_model = None

# Feature lists (derived from the notebook)
# For High Priority Ticket Model (loaded_high_priority_model) in section I
# This is the complete list of features the pipeline expects for 'High_Priority' prediction.
# It is also used as the basis for auto-tagging priority prediction.
high_priority_prediction_features = [
    'CI_Name', 'CI_Cat', 'CI_Subcat', 'WBS', 'Category',
    'Alert_Status', 'RFC_Flag',
    'No_of_Reassignments', 'No_of_Related_Interactions',
    'No_of_Related_Incidents', 'No_of_Related_Changes',
    'Reopened_Flag',
    'Open_Year', 'Open_Month', 'Open_Hour', 'Open_DayOfWeek'
]

# For Auto-Tagging - Department Model (loaded_department_model) in section III
department_features = [
    'Impact', 'Urgency', 'Category', 'CI_Subcat', 'Open_DayOfWeek',
    'Open_Hour', 'RFC_Flag', 'Alert_Status'
]

# For RFC Rule Model (loaded_rfc_rule_model) in section IV
rfc_rule_features = [
    'High_Priority', 'Impact', 'Priority',
    'No_of_Related_Incidents', 'No_of_Related_Changes'
]

# For Failure Model (loaded_failure_model) in section IV
failure_features = [
    'CI_Cat', 'CI_Subcat', 'WBS', 'Status', 'Impact', 'Urgency', 'Priority',
    'number_cnt', 'Category', 'Alert_Status', 'No_of_Reassignments',
    'No_of_Related_Interactions', 'No_of_Related_Incidents',
    'No_of_Related_Changes', 'Is_Open', 'Open_Year', 'Open_Month',
    'Open_Hour', 'Open_DayOfWeek', 'High_Priority'
]

# --- Model Loading ---
def load_all_models():
    """Loads all models into global variables."""
    global loaded_department_model
    global loaded_priority_threshold
    global loaded_high_priority_model
    global loaded_incident_volume_model
    global loaded_rfc_rule_model
    global loaded_failure_model

    try:
        app.logger.info("Loading models...")
        loaded_department_model = joblib.load('final_department_model.pkl')
        loaded_priority_threshold = joblib.load('priority_threshold.pkl')
        loaded_high_priority_model = joblib.load('high_priority_ticket_model.pkl')
        loaded_incident_volume_model = joblib.load('incident_volume_forecasting_model.pkl')
        loaded_rfc_rule_model = joblib.load('rfc_rule_model.pkl')
        loaded_failure_model = joblib.load('failure_model.pkl')
        app.logger.info("All models loaded successfully!")
    except Exception as e:
        app.logger.error(f"Error loading models: {e}")
        raise # Re-raise to indicate a critical error


# --- Helper function for data validation and preprocessing ---
def validate_and_prepare_data(raw_data_dict, required_features_list):
    """
    Validates and prepares input dictionary data into a DataFrame for model inference.
    Ensures all required features are present, adding NaNs if missing so pipelines can handle.
    """
    df = pd.DataFrame([raw_data_dict])

    # Ensure all required features are present and handle potential missing ones
    for feature in required_features_list:
        if feature not in df.columns:
            df[feature] = np.nan # Let the pipeline's imputer handle it

    # Ensure correct order and selection of features for the model's preprocessor
    df = df[required_features_list].copy()

    # Special handling for 'Impact' and 'Priority' as they can be numerical-like strings
    # and might be expected as objects or specific types by models/pipelines
    if 'Impact' in df.columns:
        df['Impact'] = df['Impact'].astype(str)
    if 'Urgency' in df.columns:
        df['Urgency'] = df['Urgency'].astype(str)
    if 'Priority' in df.columns:
        df['Priority'] = df['Priority'].astype(str)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype(str)
    if 'Alert_Status' in df.columns:
        df['Alert_Status'] = df['Alert_Status'].astype(str)
    if 'CI_Cat' in df.columns:
        df['CI_Cat'] = df['CI_Cat'].astype(str)
    if 'CI_Subcat' in df.columns:
        df['CI_Subcat'] = df['CI_Subcat'].astype(str)
    if 'WBS' in df.columns:
        df['WBS'] = df['WBS'].astype(str)
    if 'Status' in df.columns:
        df['Status'] = df['Status'].astype(str)
    if 'KB_number' in df.columns:
        df['KB_number'] = df['KB_number'].astype(str)

    # Convert numeric-like columns to numeric if applicable, handling errors
    numeric_columns_to_convert = [
        'number_cnt', 'No_of_Reassignments', 'No_of_Related_Interactions',
        'No_of_Related_Incidents', 'No_of_Related_Changes', 'Is_Open',
        'Open_Year', 'Open_Month', 'Open_Hour', 'Open_DayOfWeek', 'High_Priority',
        'RFC_Flag', 'Reopened_Flag'
    ]
    for col in numeric_columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Imputation for numeric NaNs should be handled by the pipeline's SimpleImputer.

    return df


# --- API Endpoints ---

@app.route('/predict_autotag', methods=['POST'])
def predict_autotag():
    """
    Endpoint for auto-tagging tickets with predicted department and priority.
    Takes ticket features as JSON input.
    """
    if loaded_department_model is None or loaded_high_priority_model is None or loaded_priority_threshold is None:
        return jsonify({"error": "Models not loaded. Server might be starting up or encountered an error."}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided. Please provide ticket features in JSON format."}), 400

    try:
        # Prepare data for department prediction
        dept_input_df = validate_and_prepare_data(data, department_features)

        # Make department prediction
        dept_probs = loaded_department_model.predict_proba(dept_input_df)
        predicted_department = loaded_department_model.classes_[dept_probs.argmax(axis=1)][0]
        department_confidence = float(dept_probs.max(axis=1)[0])

        # Prepare data for priority prediction using the full high_priority_prediction_features
        prio_input_df = validate_and_prepare_data(data, high_priority_prediction_features)

        # Make priority prediction
        prio_probs = loaded_high_priority_model.predict_proba(prio_input_df)[:, 1]
        predicted_priority = (prio_probs[0] > loaded_priority_threshold).astype(int)
        priority_confidence = float(prio_probs[0])

        response = {
            "predicted_department": predicted_department,
            "department_confidence": department_confidence,
            "dept_uncertain": department_confidence < 0.60, # uncertainty rule
            "predicted_priority": int(predicted_priority),
            "priority_confidence": priority_confidence
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in /predict_autotag: {e}")
        return jsonify({"error": f"Error during auto-tagging prediction: {str(e)}"}), 400

@app.route('/predict_rfc', methods=['POST'])
def predict_rfc():
    """
    Endpoint for rule-based RFC prediction.
    Takes relevant incident features as JSON input.
    """
    if loaded_rfc_rule_model is None:
        return jsonify({"error": "RFC Rule Model not loaded. Server might be starting up or encountered an error."}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided. Please provide incident features in JSON format."}), 400

    try:
        input_df = validate_and_prepare_data(data, rfc_rule_features)

        # Make RFC prediction using the rule-based model
        predicted_rfc = loaded_rfc_rule_model.predict(input_df).iloc[0]

        response = {
            "predicted_rfc": int(predicted_rfc)
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in /predict_rfc: {e}")
        return jsonify({"error": f"Error during RFC prediction: {str(e)}"}), 400

@app.route('/predict_failure', methods=['POST'])
def predict_failure():
    """
    Endpoint for ML-based failure prediction.
    Takes incident features as JSON input.
    """
    if loaded_failure_model is None:
        return jsonify({"error": "Failure Model not loaded. Server might be starting up or encountered an error."}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided. Please provide incident features in JSON format."}), 400

    try:
        input_df = validate_and_prepare_data(data, failure_features)

        # Make failure prediction
        failure_probs = loaded_failure_model.predict_proba(input_df)[:, 1]
        predicted_failure = loaded_failure_model.predict(input_df)[0]
        failure_confidence = float(failure_probs[0])

        response = {
            "predicted_failure": int(predicted_failure),
            "failure_confidence": failure_confidence
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in /predict_failure: {e}")
        return jsonify({"error": f"Error during failure prediction: {str(e)}"}), 400

@app.route('/forecast_incident_volume', methods=['POST']) # Changed to POST to allow sending 'steps' in body
def forecast_incident_volume():
    """
    Endpoint for forecasting incident volume.
    Takes an optional 'steps' parameter for the number of future periods.
    """
    if loaded_incident_volume_model is None:
        return jsonify({"error": "Incident Volume Forecasting Model not loaded. Server might be starting up or encountered an error."}), 500

    data = request.get_json(force=True)
    steps = data.get('steps', 12) if data else 12 # Default to 12 steps if no data or 'steps' not provided

    if not isinstance(steps, int) or steps <= 0:
        return jsonify({"error": "Invalid 'steps' parameter. Must be a positive integer."}), 400

    try:
        # The ARIMA model requires `steps` for forecasting
        forecast_result = loaded_incident_volume_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()

        forecast_data = []
        for i in range(steps):
            forecast_data.append({
                "date": str(forecast_mean.index[i].date()),
                "incident_count": float(forecast_mean.iloc[i]),
                "lower_ci": float(forecast_ci.iloc[i, 0]),
                "upper_ci": float(forecast_ci.iloc[i, 1])
            })

        return jsonify({"incident_volume_forecast": forecast_data})

    except Exception as e:
        app.logger.error(f"Error forecasting incident volume: {e}")
        return jsonify({"error": f"Error during incident volume forecasting: {e}"}), 400


# Call this once at the script level to load models when the app starts
load_all_models()

if __name__ = "__main__":
    print("Starting prediction API with preprocessing and model inference. . . .")
    app.run(debug=True)
