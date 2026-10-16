"""
Agri-Predict Flask Backend — app.py
Serves ML inference for Regression, Classification, and Clustering.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Artifact loading ───────────────────────────────────────────────────────────
artifacts = {}
artifact_errors = {}

def load_artifacts():
    specs = {
        'regression_pipeline': 'regression_pipeline.joblib',
        'regression_metadata': 'regression_metadata.json',
        'classification_pipeline': 'classification_pipeline.joblib',
        'classification_metadata': 'classification_metadata.json',
        'clustering_scaler': 'clustering_scaler.joblib',
        'kmeans_model': 'kmeans_model.joblib',
        'clustering_metadata': 'clustering_metadata.json',
    }
    for key, fname in specs.items():
        path = os.path.join(MODELS_DIR, fname)
        try:
            if fname.endswith('.json'):
                with open(path, 'r') as f:
                    artifacts[key] = json.load(f)
            else:
                artifacts[key] = joblib.load(path)
            print(f"  [OK] Loaded {fname}")
        except FileNotFoundError:
            artifact_errors[key] = f"Missing: {path}"
            print(f"  [MISSING] {fname}")
        except Exception as e:
            artifact_errors[key] = str(e)
            print(f"  [ERROR] {fname}: {e}")

print("Loading model artifacts...")
load_artifacts()

# ── Validation helpers ─────────────────────────────────────────────────────────
def validate_numeric(value, name, lo=None, hi=None):
    """Return (float_value, None) or (None, error_str)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, f"'{name}' must be a number, got {repr(value)}"
    if not np.isfinite(v):
        return None, f"'{name}' must be finite"
    if lo is not None and v < lo:
        return None, f"'{name}' must be >= {lo} (got {v})"
    if hi is not None and v > hi:
        return None, f"'{name}' must be <= {hi} (got {v})"
    return v, None

def require_artifact(key):
    if key not in artifacts:
        return jsonify({'error': f'Model artifact not available: {key}. '
                                 f'Run export_models.py to generate it.'}), 503
    return None

# ── Static frontend serving ────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

# ── /api/health ────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    status = {
        'status': 'ok' if not artifact_errors else 'degraded',
        'artifacts': {
            key: ('loaded' if key in artifacts else 'missing')
            for key in [
                'regression_pipeline', 'classification_pipeline',
                'kmeans_model', 'clustering_scaler', 'clustering_metadata'
            ]
        },
        'errors': artifact_errors or None
    }
    code = 200 if status['status'] == 'ok' else 206
    return jsonify(status), code

# ── /api/metadata ──────────────────────────────────────────────────────────────
@app.route('/api/metadata')
def metadata():
    result = {}
    if 'regression_metadata' in artifacts:
        m = artifacts['regression_metadata']
        result['regression'] = {
            'model': m['model_type'],
            'features': m['features'],
            'target': m['target'],
            'metrics': m['metrics']
        }
    if 'classification_metadata' in artifacts:
        m = artifacts['classification_metadata']
        result['classification'] = {
            'model': m['model_type'],
            'features': m['features'],
            'feature_units': m['feature_units'],
            'classes': m['classes'],
            'metrics': m['metrics']
        }
    if 'clustering_metadata' in artifacts:
        m = artifacts['clustering_metadata']
        result['clustering'] = {
            'algorithm': m['algorithm'],
            'n_clusters': m['n_clusters'],
            'features': m['features'],
            'cluster_sizes': m['cluster_sizes'],
            'metrics': m['metrics'],
            'agglomerative_predict_supported': m['agglomerative_predict_supported']
        }
    return jsonify(result)

# ── /api/regression/predict ────────────────────────────────────────────────────
REGRESSION_FIELDS = {
    'Crop_Year': (1990, 2030),
    'Area': (0.1, 1e7),
    'Annual_Rainfall': (0, 5000),
    'Fertilizer': (0, 1e7),
    'Pesticide': (0, 1e6),
    'Avg_Temperature': (-20, 60),
    'Max_Temperature': (-20, 70),
    'Min_Temperature': (-30, 60),
}
REGRESSION_CAT = ['Crop', 'Season', 'State']

@app.route('/api/regression/predict', methods=['POST'])
def regression_predict():
    err_resp = require_artifact('regression_pipeline')
    if err_resp: return err_resp

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    errors = {}

    # Validate categoricals
    cat_values = {}
    for col in REGRESSION_CAT:
        val = data.get(col, '').strip()
        if not val:
            errors[col] = f"'{col}' is required"
        else:
            cat_values[col] = val

    # Validate numerics
    num_values = {}
    for col, (lo, hi) in REGRESSION_FIELDS.items():
        v, err = validate_numeric(data.get(col), col, lo, hi)
        if err:
            errors[col] = err
        else:
            num_values[col] = v

    # Derived feature
    if 'Max_Temperature' in num_values and 'Min_Temperature' in num_values:
        num_values['Temperature_Range'] = num_values['Max_Temperature'] - num_values['Min_Temperature']
    elif not errors:
        errors['Temperature_Range'] = "Max_Temperature and Min_Temperature required to compute Temperature_Range"

    if errors:
        return jsonify({'error': 'Validation failed', 'fields': errors}), 422

    row = {**cat_values, **num_values}
    col_order = ['Crop', 'Season', 'State', 'Crop_Year', 'Area', 'Annual_Rainfall',
                 'Fertilizer', 'Pesticide', 'Avg_Temperature', 'Max_Temperature',
                 'Min_Temperature', 'Temperature_Range']
    X_in = pd.DataFrame([{c: row[c] for c in col_order}])

    try:
        pred = float(artifacts['regression_pipeline'].predict(X_in)[0])
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

    if not np.isfinite(pred):
        return jsonify({'error': 'Model returned non-finite prediction'}), 500

    return jsonify({
        'operation': 'regression',
        'predicted_yield': round(pred, 2),
        'unit': 'kg per unit area',
        'model': 'Gradient Boosting Regressor (Tuned)',
        'model_r2': artifacts.get('regression_metadata', {}).get('metrics', {}).get('r2'),
        'model_rmse': artifacts.get('regression_metadata', {}).get('metrics', {}).get('rmse'),
        'inputs_received': row
    })

# ── /api/classification/predict ───────────────────────────────────────────────
CLASSIFICATION_FIELDS = {
    'N': (0, 300),
    'P': (0, 300),
    'K': (0, 300),
    'temperature': (-10, 55),
    'humidity': (0, 100),
    'ph': (0, 14),
    'rainfall': (0, 3000),
}

@app.route('/api/classification/predict', methods=['POST'])
def classification_predict():
    err_resp = require_artifact('classification_pipeline')
    if err_resp: return err_resp

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    errors = {}
    values = {}
    for col, (lo, hi) in CLASSIFICATION_FIELDS.items():
        v, err = validate_numeric(data.get(col), col, lo, hi)
        if err:
            errors[col] = err
        else:
            values[col] = v

    if errors:
        return jsonify({'error': 'Validation failed', 'fields': errors}), 422

    feat_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_in = pd.DataFrame([[values[f] for f in feat_order]], columns=feat_order)

    try:
        pipe = artifacts['classification_pipeline']
        pred_class = pipe.predict(X_in)[0]
        probs = pipe.predict_proba(X_in)[0]
        classes = pipe.classes_.tolist()
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

    idx_sorted = np.argsort(probs)[::-1]
    top3 = [
        {'crop': classes[i], 'probability': round(float(probs[i]), 4)}
        for i in idx_sorted[:3]
    ]

    return jsonify({
        'operation': 'classification',
        'recommended_crop': pred_class,
        'top_3_candidates': top3,
        'n_classes': len(classes),
        'model': 'Random Forest Classifier (Tuned, k=4 CV)',
        'model_accuracy': artifacts.get('classification_metadata', {}).get('metrics', {}).get('accuracy'),
        'probability_support': True,
        'inputs_received': values
    })

# ── /api/clustering/assign ────────────────────────────────────────────────────
CLUSTERING_FIELDS = {
    'soil_ph': (3.0, 10.0),
    'organic_matter': (0.0, 20.0),
    'sand_pct': (0.0, 100.0),
    'temperature': (-10.0, 55.0),
    'humidity': (0.0, 100.0),
    'rainfall': (0.0, 3000.0),
    'ndvi': (0.0, 1.0),
}

@app.route('/api/clustering/assign', methods=['POST'])
@app.route('/api/clustering/predict', methods=['POST'])
def clustering_assign():
    err_resp = require_artifact('kmeans_model')
    if err_resp: return err_resp
    err_resp = require_artifact('clustering_scaler')
    if err_resp: return err_resp
    err_resp = require_artifact('clustering_metadata')
    if err_resp: return err_resp

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    errors = {}
    values = {}
    for col, (lo, hi) in CLUSTERING_FIELDS.items():
        v, err = validate_numeric(data.get(col), col, lo, hi)
        if err:
            errors[col] = err
        else:
            values[col] = v

    if errors:
        return jsonify({'error': 'Validation failed', 'fields': errors}), 422

    feat_order = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature',
                  'humidity', 'rainfall', 'ndvi']
    X_in = pd.DataFrame([[values[f] for f in feat_order]], columns=feat_order)

    try:
        X_scaled = artifacts['clustering_scaler'].transform(X_in)
        cluster_label = int(artifacts['kmeans_model'].predict(X_scaled)[0])
    except Exception as e:
        return jsonify({'error': f'Clustering inference failed: {str(e)}'}), 500

    meta = artifacts['clustering_metadata']
    profile = meta['cluster_profiles'].get(str(cluster_label), {})
    cluster_size = meta['cluster_sizes'].get(str(cluster_label), 'N/A')

    profile_summary = {
        feat: {'mean': profile[feat]['mean'], 'std': profile[feat]['std']}
        for feat in feat_order
        if feat in profile
    }
    yield_profile = profile.get('yield', {})

    return jsonify({
        'operation': 'clustering',
        'assigned_cluster': cluster_label,
        'cluster_label': f'Zone {cluster_label}',
        'cluster_size': cluster_size,
        'cluster_profile': profile_summary,
        'cluster_mean_yield_kgha': yield_profile.get('mean'),
        'algorithm': 'K-Means (k=4)',
        'agglomerative_note': (
            'Agglomerative Hierarchical Clustering is used for training-data exploration only. '
            'New-field assignment uses K-Means, which supports native .predict().'
        ),
        'exploratory_note': (
            'This is an exploratory environmental zone assignment. '
            'Cluster boundaries reflect gradients in soil and climate conditions '
            'and should not be interpreted as categorical crop suitability scores.'
        ),
        'inputs_received': values
    })

# ── /api/analysis/complete ────────────────────────────────────────────────────
@app.route('/api/analysis/complete', methods=['POST'])
def complete_analysis():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    result = {}

    # --- Regression ---
    reg_data = data.get('regression', {})
    reg_errors = {}
    reg_num_values = {}
    reg_cat_values = {}

    for col in REGRESSION_CAT:
        val = str(reg_data.get(col, '')).strip()
        if not val:
            reg_errors[col] = f"'{col}' is required"
        else:
            reg_cat_values[col] = val

    for col, (lo, hi) in REGRESSION_FIELDS.items():
        v, err = validate_numeric(reg_data.get(col), col, lo, hi)
        if err:
            reg_errors[col] = err
        else:
            reg_num_values[col] = v

    if 'Max_Temperature' in reg_num_values and 'Min_Temperature' in reg_num_values:
        reg_num_values['Temperature_Range'] = (
            reg_num_values['Max_Temperature'] - reg_num_values['Min_Temperature']
        )

    if reg_errors:
        result['regression'] = {'status': 'error', 'errors': reg_errors}
    elif 'regression_pipeline' not in artifacts:
        result['regression'] = {'status': 'unavailable', 'error': 'Regression model not loaded'}
    else:
        try:
            row = {**reg_cat_values, **reg_num_values}
            col_order = ['Crop', 'Season', 'State', 'Crop_Year', 'Area', 'Annual_Rainfall',
                         'Fertilizer', 'Pesticide', 'Avg_Temperature', 'Max_Temperature',
                         'Min_Temperature', 'Temperature_Range']
            X_in = pd.DataFrame([{c: row[c] for c in col_order}])
            pred = float(artifacts['regression_pipeline'].predict(X_in)[0])
            result['regression'] = {
                'status': 'success',
                'predicted_yield': round(pred, 2),
                'unit': 'kg per unit area'
            }
        except Exception as e:
            result['regression'] = {'status': 'error', 'error': str(e)}

    # --- Classification ---
    cls_data = data.get('classification', {})
    cls_errors = {}
    cls_values = {}

    for col, (lo, hi) in CLASSIFICATION_FIELDS.items():
        v, err = validate_numeric(cls_data.get(col), col, lo, hi)
        if err:
            cls_errors[col] = err
        else:
            cls_values[col] = v

    if cls_errors:
        result['classification'] = {'status': 'error', 'errors': cls_errors}
    elif 'classification_pipeline' not in artifacts:
        result['classification'] = {'status': 'unavailable', 'error': 'Classification model not loaded'}
    else:
        try:
            feat_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            X_in = pd.DataFrame([[cls_values[f] for f in feat_order]], columns=feat_order)
            pipe = artifacts['classification_pipeline']
            pred_class = pipe.predict(X_in)[0]
            probs = pipe.predict_proba(X_in)[0]
            classes = pipe.classes_.tolist()
            top3 = [
                {'crop': classes[i], 'probability': round(float(probs[i]), 4)}
                for i in np.argsort(probs)[::-1][:3]
            ]
            result['classification'] = {
                'status': 'success',
                'recommended_crop': pred_class,
                'top_3_candidates': top3
            }
        except Exception as e:
            result['classification'] = {'status': 'error', 'error': str(e)}

    # --- Clustering ---
    clu_data = data.get('clustering', {})
    clu_errors = {}
    clu_values = {}

    for col, (lo, hi) in CLUSTERING_FIELDS.items():
        v, err = validate_numeric(clu_data.get(col), col, lo, hi)
        if err:
            clu_errors[col] = err
        else:
            clu_values[col] = v

    if clu_errors:
        result['clustering'] = {'status': 'error', 'errors': clu_errors}
    elif 'kmeans_model' not in artifacts or 'clustering_scaler' not in artifacts:
        result['clustering'] = {'status': 'unavailable', 'error': 'Clustering models not loaded'}
    else:
        try:
            feat_order = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature',
                          'humidity', 'rainfall', 'ndvi']
            X_in = pd.DataFrame([[clu_values[f] for f in feat_order]], columns=feat_order)
            X_scaled = artifacts['clustering_scaler'].transform(X_in)
            cluster_label = int(artifacts['kmeans_model'].predict(X_scaled)[0])
            meta = artifacts['clustering_metadata']
            yield_profile = meta['cluster_profiles'].get(str(cluster_label), {}).get('yield', {})
            result['clustering'] = {
                'status': 'success',
                'assigned_cluster': cluster_label,
                'cluster_label': f'Zone {cluster_label}',
                'cluster_mean_yield_kgha': yield_profile.get('mean')
            }
        except Exception as e:
            result['clustering'] = {'status': 'error', 'error': str(e)}

    overall = 'success' if all(
        result.get(k, {}).get('status') == 'success'
        for k in ['regression', 'classification', 'clustering']
    ) else 'partial'

    return jsonify({'operation': 'complete_analysis', 'status': overall, 'results': result})


if __name__ == '__main__':
    print(f"\nAgri-Predict backend starting...")
    print(f"Frontend: {FRONTEND_DIR}")
    print(f"Models:   {MODELS_DIR}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
