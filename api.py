import os
import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from assets_data_prep import prepare_data

# ─── טעינת המודל פעם אחת בזמן הפעלת השרת ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pkl')

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. "
                       "Make sure trained_model.pkl is in the project root.")

app = Flask(__name__)


# ─── GET / — הגשת דף ה-HTML ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─── POST /predict — קבלת נתונים, עיבוד, חיזוי ──────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    # 1. קבלת JSON מהטופס
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Invalid JSON payload.'}), 400

    # 2. שליפה וולידציה של שדות חובה / אופציונליים
    errors = {}

    # startYear — חובה, חייב להיות מספר
    raw_year = payload.get('startYear', '')
    if raw_year == '' or raw_year is None:
        start_year = np.nan          # NaN חוקי — ה-Imputer יטפל בו
    else:
        try:
            start_year = float(raw_year)
        except (ValueError, TypeError):
            errors['startYear'] = 'חייב להיות מספר שנה (למשל: 2010).'

    # runtimeMinutes — אופציונלי, אם קיים חייב להיות מספר
    raw_runtime = payload.get('runtimeMinutes', '')
    if raw_runtime == '' or raw_runtime is None:
        runtime = np.nan
    else:
        try:
            runtime = float(raw_runtime)
        except (ValueError, TypeError):
            errors['runtimeMinutes'] = 'חייב להיות מספר (למשל: 120).'

    if errors:
        return jsonify({'error': 'שגיאת קלט', 'fields': errors}), 400

    # 3. בניית DataFrame של שורה אחת — בדיוק כמו הדאטה המקורי
    row = {
        'startYear':       start_year,
        'runtimeMinutes':  runtime,
        'genres':          payload.get('genres', ''),
        'lead_actors_ids': payload.get('lead_actors_ids', ''),
        'Country':         payload.get('Country', ''),
        'Language':        payload.get('Language', ''),
        'budget':          payload.get('budget', None) or None,
        'plot':            payload.get('plot', None) or None,
    }
    df_input = pd.DataFrame([row])

    # 4. עיבוד — prepare_data מחזירה 36 פיצ'רים בסדר הנכון
    try:
        df_processed = prepare_data(df_input)
    except Exception as e:
        return jsonify({'error': f'שגיאה בעיבוד הנתונים: {str(e)}'}), 500

    # 5. חיזוי
    try:
        prediction = float(model.predict(df_processed)[0])
        prediction = round(max(1.0, min(10.0, prediction)), 2)
        print(f"[PREDICT] קלט: שנה={row['startYear']}, ז'אנרים={row['genres']}, מדינה={row['Country']} → ציון חזוי: {prediction}")
    except Exception as e:
        return jsonify({'error': f'שגיאה בחיזוי: {str(e)}'}), 500

    return jsonify({'predicted_rating': prediction})


# ─── הרצה ישירה (פיתוח בלבד) ─────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
