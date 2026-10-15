/**
 * api.js — Agri-Predict API client
 * All fetch calls to the Flask backend. Falls back to localhost:5000.
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? `http://${window.location.hostname}:5000`
  : '';

const Api = (() => {

  async function _post(endpoint, payload) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      const msg = data.error || `HTTP ${res.status}`;
      const fields = data.fields ? ` — ${JSON.stringify(data.fields)}` : '';
      throw new Error(msg + fields);
    }
    return data;
  }

  async function _get(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    return res.json();
  }

  return {
    health: () => _get('/api/health'),
    metadata: () => _get('/api/metadata'),
    regression: (payload) => _post('/api/regression/predict', payload),
    classification: (payload) => _post('/api/classification/predict', payload),
    clustering: (payload) => _post('/api/clustering/assign', payload),
    completeAnalysis: (payload) => _post('/api/analysis/complete', payload),
  };
})();

window.Api = Api;
