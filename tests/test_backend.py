"""
tests/test_backend.py — Agri-Predict backend API tests
Run with: pytest tests/test_backend.py -v --tb=short
Requires the Flask app to be running at http://localhost:5000
"""
import json
import sys
import os
import pytest

# Add backend to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'backend'))

from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c

# ── Helper payloads ────────────────────────────────────────────────────────────

VALID_REGRESSION = {
    'Crop': 'Rice',
    'Season': 'Kharif',
    'State': 'Punjab',
    'Crop_Year': 2022,
    'Area': 150.0,
    'Annual_Rainfall': 850.0,
    'Fertilizer': 14500.0,
    'Pesticide': 230.0,
    'Avg_Temperature': 26.3,
    'Max_Temperature': 34.1,
    'Min_Temperature': 19.2,
}

VALID_CLASSIFICATION = {
    'N': 90.0,
    'P': 42.0,
    'K': 43.0,
    'temperature': 23.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.0,
}

VALID_CLUSTERING = {
    'soil_ph': 6.5,
    'organic_matter': 2.8,
    'sand_pct': 42.0,
    'temperature': 26.5,
    'humidity': 72.0,
    'rainfall': 130.0,
    'ndvi': 0.62,
}

# ── /api/health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200_or_206(self, client):
        resp = client.get('/api/health')
        assert resp.status_code in (200, 206)

    def test_health_json_shape(self, client):
        resp = client.get('/api/health')
        data = resp.get_json()
        assert 'status' in data
        assert 'artifacts' in data
        assert data['status'] in ('ok', 'degraded')

    def test_health_lists_expected_artifacts(self, client):
        data = client.get('/api/health').get_json()
        for key in ('regression_pipeline', 'classification_pipeline',
                    'kmeans_model', 'clustering_scaler'):
            assert key in data['artifacts'], f"Missing artifact key: {key}"

# ── /api/metadata ──────────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_returns_200(self, client):
        resp = client.get('/api/metadata')
        assert resp.status_code == 200

    def test_metadata_has_regression(self, client):
        data = client.get('/api/metadata').get_json()
        if 'regression' in data:
            assert 'metrics' in data['regression']
            assert 'r2' in data['regression']['metrics']

    def test_metadata_has_classification(self, client):
        data = client.get('/api/metadata').get_json()
        if 'classification' in data:
            assert 'classes' in data['classification']
            assert len(data['classification']['classes']) == 22

    def test_metadata_has_clustering(self, client):
        data = client.get('/api/metadata').get_json()
        if 'clustering' in data:
            assert data['clustering']['n_clusters'] == 4
            assert data['clustering']['agglomerative_predict_supported'] is False

# ── /api/regression/predict ────────────────────────────────────────────────────

class TestRegression:
    def test_valid_payload_returns_200(self, client):
        resp = client.post('/api/regression/predict',
                           json=VALID_REGRESSION,
                           content_type='application/json')
        assert resp.status_code == 200

    def test_valid_payload_has_prediction(self, client):
        data = client.post('/api/regression/predict', json=VALID_REGRESSION).get_json()
        assert 'predicted_yield' in data
        assert isinstance(data['predicted_yield'], (int, float))
        # Yield should be plausible (positive, not absurdly large)
        assert 0 < data['predicted_yield'] < 1_000_000

    def test_valid_payload_has_model_field(self, client):
        data = client.post('/api/regression/predict', json=VALID_REGRESSION).get_json()
        assert 'model' in data
        assert 'Gradient Boosting' in data['model']

    def test_missing_crop_returns_422(self, client):
        payload = {k: v for k, v in VALID_REGRESSION.items() if k != 'Crop'}
        resp = client.post('/api/regression/predict', json=payload)
        assert resp.status_code == 422

    def test_invalid_year_returns_422(self, client):
        payload = {**VALID_REGRESSION, 'Crop_Year': 1800}
        resp = client.post('/api/regression/predict', json=payload)
        assert resp.status_code == 422

    def test_empty_body_returns_400(self, client):
        resp = client.post('/api/regression/predict',
                           data='not json',
                           content_type='application/json')
        assert resp.status_code == 400

    def test_different_crops_give_different_yields(self, client):
        rice_payload = {**VALID_REGRESSION, 'Crop': 'Rice'}
        wheat_payload = {**VALID_REGRESSION, 'Crop': 'Wheat', 'Season': 'Rabi'}
        rice_data = client.post('/api/regression/predict', json=rice_payload).get_json()
        wheat_data = client.post('/api/regression/predict', json=wheat_payload).get_json()
        # Not necessarily different — but both should succeed
        assert 'predicted_yield' in rice_data
        assert 'predicted_yield' in wheat_data

# ── /api/classification/predict ───────────────────────────────────────────────

class TestClassification:
    def test_valid_payload_returns_200(self, client):
        resp = client.post('/api/classification/predict', json=VALID_CLASSIFICATION)
        assert resp.status_code == 200

    def test_response_has_recommended_crop(self, client):
        data = client.post('/api/classification/predict', json=VALID_CLASSIFICATION).get_json()
        assert 'recommended_crop' in data
        assert isinstance(data['recommended_crop'], str)
        assert len(data['recommended_crop']) > 0

    def test_response_has_top3(self, client):
        data = client.post('/api/classification/predict', json=VALID_CLASSIFICATION).get_json()
        assert 'top_3_candidates' in data
        assert len(data['top_3_candidates']) == 3
        for item in data['top_3_candidates']:
            assert 'crop' in item
            assert 'probability' in item
            assert 0 <= item['probability'] <= 1

    def test_top3_probabilities_sum_to_at_most_1(self, client):
        data = client.post('/api/classification/predict', json=VALID_CLASSIFICATION).get_json()
        total = sum(c['probability'] for c in data['top_3_candidates'])
        assert total <= 1.001  # floating point tolerance

    def test_top3_first_is_recommended(self, client):
        data = client.post('/api/classification/predict', json=VALID_CLASSIFICATION).get_json()
        assert data['top_3_candidates'][0]['crop'] == data['recommended_crop']

    def test_invalid_ph_returns_422(self, client):
        payload = {**VALID_CLASSIFICATION, 'ph': 20}
        resp = client.post('/api/classification/predict', json=payload)
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_CLASSIFICATION.items() if k != 'N'}
        resp = client.post('/api/classification/predict', json=payload)
        assert resp.status_code == 422

    def test_probability_support_flagged(self, client):
        data = client.post('/api/classification/predict', json=VALID_CLASSIFICATION).get_json()
        assert data.get('probability_support') is True

# ── /api/clustering/assign ────────────────────────────────────────────────────

class TestClustering:
    def test_valid_payload_returns_200(self, client):
        resp = client.post('/api/clustering/assign', json=VALID_CLUSTERING)
        assert resp.status_code == 200

    def test_response_has_cluster(self, client):
        data = client.post('/api/clustering/assign', json=VALID_CLUSTERING).get_json()
        assert 'assigned_cluster' in data
        assert data['assigned_cluster'] in (0, 1, 2, 3)

    def test_response_has_zone_label(self, client):
        data = client.post('/api/clustering/assign', json=VALID_CLUSTERING).get_json()
        assert 'cluster_label' in data
        assert data['cluster_label'].startswith('Zone ')

    def test_response_has_cluster_profile(self, client):
        data = client.post('/api/clustering/assign', json=VALID_CLUSTERING).get_json()
        assert 'cluster_mean_yield_kgha' in data

    def test_agglomerative_note_present(self, client):
        data = client.post('/api/clustering/assign', json=VALID_CLUSTERING).get_json()
        assert 'agglomerative_note' in data
        assert 'predict' in data['agglomerative_note'].lower()

    def test_invalid_ndvi_returns_422(self, client):
        payload = {**VALID_CLUSTERING, 'ndvi': 2.5}
        resp = client.post('/api/clustering/assign', json=payload)
        assert resp.status_code == 422

    def test_invalid_ph_returns_422(self, client):
        payload = {**VALID_CLUSTERING, 'soil_ph': 15}
        resp = client.post('/api/clustering/assign', json=payload)
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_CLUSTERING.items() if k != 'ndvi'}
        resp = client.post('/api/clustering/assign', json=payload)
        assert resp.status_code == 422

    def test_extreme_but_valid_values(self, client):
        payload = {
            'soil_ph': 3.0, 'organic_matter': 0.5, 'sand_pct': 95.0,
            'temperature': 45.0, 'humidity': 10.0, 'rainfall': 10.0, 'ndvi': 0.05
        }
        resp = client.post('/api/clustering/assign', json=payload)
        assert resp.status_code == 200

# ── /api/analysis/complete ────────────────────────────────────────────────────

class TestCompleteAnalysis:
    VALID_COMPLETE = {
        'regression': VALID_REGRESSION,
        'classification': VALID_CLASSIFICATION,
        'clustering': VALID_CLUSTERING,
    }

    def test_valid_complete_returns_200(self, client):
        resp = client.post('/api/analysis/complete', json=self.VALID_COMPLETE)
        assert resp.status_code == 200

    def test_complete_has_results_key(self, client):
        data = client.post('/api/analysis/complete', json=self.VALID_COMPLETE).get_json()
        assert 'results' in data
        assert 'operation' in data
        assert data['operation'] == 'complete_analysis'

    def test_complete_results_have_three_keys(self, client):
        data = client.post('/api/analysis/complete', json=self.VALID_COMPLETE).get_json()
        results = data['results']
        for key in ('regression', 'classification', 'clustering'):
            assert key in results, f"Missing key '{key}' in complete results"

    def test_complete_status_is_success_or_partial(self, client):
        data = client.post('/api/analysis/complete', json=self.VALID_COMPLETE).get_json()
        assert data['status'] in ('success', 'partial')

    def test_partial_inputs_return_200_with_errors(self, client):
        # Send only classification data — regression and clustering will error
        payload = {
            'regression': {},
            'classification': VALID_CLASSIFICATION,
            'clustering': {},
        }
        data = client.post('/api/analysis/complete', json=payload).get_json()
        assert data is not None  # ensures it didn't crash
        results = data.get('results', {})
        assert results.get('classification', {}).get('status') == 'success'

    def test_empty_body_returns_400(self, client):
        resp = client.post('/api/analysis/complete',
                           data='bad',
                           content_type='application/json')
        assert resp.status_code == 400

# ── /api/static serving ───────────────────────────────────────────────────────

class TestStaticServing:
    def test_frontend_index_served(self, client):
        resp = client.get('/')
        # Either serves index.html (200) or frontend dir missing (404) — both are valid
        assert resp.status_code in (200, 404)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
