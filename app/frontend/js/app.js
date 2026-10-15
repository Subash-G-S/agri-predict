/**
 * app.js — Agri-Predict main application controller
 * Handles mode switching, form submission, API calls, and result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Mode definitions ────────────────────────────────────────────────────────
  const MODES = {
    regression: {
      headline: 'Crop Yield Prediction',
      sub: 'Estimate crop yield from field area, climate, and agronomic inputs using a Gradient Boosting regression model trained on 19,689 Indian agricultural records.',
      badge: 'Gradient Boosting · R² 0.9709',
    },
    classification: {
      headline: 'Crop Recommendation',
      sub: 'Identify the optimal crop for a given soil-climate condition using a Random Forest classifier trained on 2,200 labelled records across 22 crop classes.',
      badge: 'Random Forest · 99.6% accuracy',
    },
    clustering: {
      headline: 'Maize Field Zone Assignment',
      sub: 'Assign a maize field to one of four environmental zones discovered by K-Means clustering on 10,000 maize-field monitoring records.',
      badge: 'K-Means k=4 · Agglomerative (exploratory)',
    },
    complete: {
      headline: 'Full Analysis Suite',
      sub: 'Run yield prediction, crop recommendation, and field zone assignment simultaneously using all three ML models.',
      badge: 'Regression + Classification + Clustering',
    },
  };

  // ── DOM refs ────────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  // ── Health check ────────────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const data = await Api.health();
      const dot  = $('status-dot');
      const lbl  = $('status-label');
      if (data.status === 'ok') {
        dot.className = 'status-dot ok';
        lbl.textContent = 'Models loaded';
      } else {
        dot.className = 'status-dot warn';
        lbl.textContent = 'Some models missing';
      }
    } catch {
      $('status-dot').className = 'status-dot err';
      $('status-label').textContent = 'Backend offline';
    }
  }

  // ── Fetch metadata and update badge ────────────────────────────────────────
  async function loadMetadata() {
    try {
      const meta = await Api.metadata();
      if (meta.regression && meta.regression.metrics) {
        const r2 = meta.regression.metrics.r2;
        MODES.regression.badge = `Gradient Boosting · R² ${r2}`;
      }
      if (meta.classification && meta.classification.metrics) {
        const acc = (meta.classification.metrics.accuracy * 100).toFixed(2);
        MODES.classification.badge = `Random Forest · ${acc}% accuracy`;
      }
    } catch { /* non-critical */ }
  }

  // ── Mode switching ──────────────────────────────────────────────────────────
  let currentMode = 'regression';

  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === currentMode) return;

      // Deactivate current
      document.getElementById(`nav-${currentMode}`).classList.remove('active');
      document.getElementById(`panel-${currentMode}`).classList.remove('active');
      document.getElementById(`panel-${currentMode}`).classList.add('hidden');

      // Activate new
      currentMode = mode;
      btn.classList.add('active');
      const panel = document.getElementById(`panel-${mode}`);
      panel.classList.remove('hidden');
      panel.classList.add('active');

      // Update hero
      const m = MODES[mode];
      $('page-headline').textContent = m.headline;
      $('page-sub').textContent = m.sub;
      $('badge-value').textContent = m.badge;
    });
  });

  // ── UI helpers ──────────────────────────────────────────────────────────────
  function showEl(...ids) { ids.forEach(id => $(`${id}`) && ($(`${id}`).classList.remove('hidden'))); }
  function hideEl(...ids) { ids.forEach(id => $(`${id}`) && ($(`${id}`).classList.add('hidden'))); }

  function setLoading(mode, isLoading) {
    if (isLoading) {
      hideEl(`result-${mode}`, `error-${mode}`);
      showEl(`loading-${mode}`);
    } else {
      hideEl(`loading-${mode}`);
    }
  }

  function showError(mode, msg) {
    hideEl(`loading-${mode}`, `result-${mode}`);
    $(`error-${mode}-msg`).textContent = msg;
    showEl(`error-${mode}`);
  }

  function collectForm(formEl) {
    const fd = new FormData(formEl);
    const data = {};
    for (const [k, v] of fd.entries()) data[k] = v;
    // Also collect named inputs not in FormData (e.g., disabled)
    formEl.querySelectorAll('[name]').forEach(el => {
      if (!(el.name in data)) data[el.name] = el.value;
    });
    return data;
  }

  // ── REGRESSION ──────────────────────────────────────────────────────────────
  const regFields = ['Crop', 'Season', 'State', 'Crop_Year', 'Area',
    'Annual_Rainfall', 'Fertilizer', 'Pesticide',
    'Avg_Temperature', 'Max_Temperature', 'Min_Temperature'];

  $('form-regression').addEventListener('submit', async (e) => {
    e.preventDefault();
    Validation.clearErrors('reg', regFields);
    const raw = collectForm(e.target);
    const errors = Validation.regression(raw);
    if (Object.keys(errors).length) {
      // Map error keys to display prefix
      const mapped = {};
      Object.entries(errors).forEach(([k, v]) => {
        const dispKey = k.replace(/_/g, '-').toLowerCase();
        // Try direct match first
        const elId = `err-reg-${dispKey}`;
        if (document.getElementById(elId)) mapped[dispKey] = v;
        else mapped[k] = v;
      });
      Validation.showErrors(mapped, 'reg');
      return;
    }

    setLoading('regression', true);
    const payload = {};
    regFields.forEach(k => {
      payload[k] = ['Crop', 'Season', 'State'].includes(k) ? raw[k] : parseFloat(raw[k]);
    });

    try {
      const result = await Api.regression(payload);
      setLoading('regression', false);
      renderRegressionResult(result);
    } catch (err) {
      setLoading('regression', false);
      showError('regression', err.message);
    }
  });

  $('btn-reg-reset').addEventListener('click', () => {
    $('form-regression').reset();
    Validation.clearErrors('reg', regFields);
    hideEl('result-regression', 'error-regression');
  });

  function renderRegressionResult(data) {
    $('reg-yield-value').textContent = data.predicted_yield.toLocaleString('en-IN', {maximumFractionDigits: 1});
    $('reg-model-tag').textContent = data.model || 'Gradient Boosting';
    $('reg-r2-value').textContent = data.model_r2 != null ? data.model_r2.toFixed(4) : '—';
    $('reg-rmse-value').textContent = data.model_rmse != null ? data.model_rmse.toFixed(2) : '—';
    $('reg-result-note').textContent =
      `Estimated yield for ${data.inputs_received.Crop || 'crop'} in `
      + `${data.inputs_received.State || 'selected state'}, `
      + `${data.inputs_received.Season || ''} season ${data.inputs_received.Crop_Year || ''}.`;
    showEl('result-regression');
  }

  // ── CLASSIFICATION ──────────────────────────────────────────────────────────
  const clsFields = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall'];

  $('form-classification').addEventListener('submit', async (e) => {
    e.preventDefault();
    Validation.clearErrors('cls', clsFields);
    const raw = collectForm(e.target);
    const errors = Validation.classification(raw);
    if (Object.keys(errors).length) {
      Validation.showErrors(errors, 'cls');
      return;
    }

    setLoading('classification', true);
    const payload = {};
    clsFields.forEach(k => { payload[k] = parseFloat(raw[k]); });

    try {
      const result = await Api.classification(payload);
      setLoading('classification', false);
      renderClassificationResult(result);
    } catch (err) {
      setLoading('classification', false);
      showError('classification', err.message);
    }
  });

  $('btn-cls-reset').addEventListener('click', () => {
    $('form-classification').reset();
    Validation.clearErrors('cls', clsFields);
    hideEl('result-classification', 'error-classification');
  });

  function renderClassificationResult(data) {
    $('cls-crop-value').textContent = data.recommended_crop;
    $('cls-model-tag').textContent = data.model || 'Random Forest';
    $('cls-accuracy').textContent = data.model_accuracy != null
      ? `${(data.model_accuracy * 100).toFixed(2)}%` : '—';

    // Top 3
    const list = $('cls-top3-list');
    list.innerHTML = '';
    (data.top_3_candidates || []).forEach((item, i) => {
      const pct = (item.probability * 100).toFixed(1);
      const el = document.createElement('div');
      el.className = 'top3-item';
      el.innerHTML = `
        <span class="top3-rank rank-${i+1}">${i+1}</span>
        <span class="top3-crop">${item.crop}</span>
        <div class="top3-bar-wrap">
          <div class="top3-bar" style="width:${pct}%"></div>
        </div>
        <span class="top3-prob">${pct}%</span>
      `;
      list.appendChild(el);
    });

    showEl('result-classification');
  }

  // ── CLUSTERING ──────────────────────────────────────────────────────────────
  const cluFields = ['soil_ph', 'organic_matter', 'sand_pct',
    'ndvi', 'temperature', 'humidity', 'rainfall'];
  const cluErrKeys = {
    'soil_ph': 'clu-soil-ph',
    'organic_matter': 'clu-organic',
    'sand_pct': 'clu-sand',
    'ndvi': 'clu-ndvi',
    'temperature': 'clu-temperature',
    'humidity': 'clu-humidity',
    'rainfall': 'clu-rainfall',
  };

  $('form-clustering').addEventListener('submit', async (e) => {
    e.preventDefault();
    cluFields.forEach(k => {
      const dispKey = cluErrKeys[k];
      const el = document.getElementById(`err-${dispKey}`);
      if (el) el.textContent = '';
    });

    const raw = collectForm(e.target);
    const errors = Validation.clustering(raw);
    if (Object.keys(errors).length) {
      Object.entries(errors).forEach(([k, v]) => {
        const dispKey = cluErrKeys[k];
        if (dispKey) {
          const el = document.getElementById(`err-${dispKey}`);
          if (el) el.textContent = v;
        }
      });
      return;
    }

    setLoading('clustering', true);
    const payload = {};
    cluFields.forEach(k => { payload[k] = parseFloat(raw[k]); });

    try {
      const result = await Api.clustering(payload);
      setLoading('clustering', false);
      renderClusteringResult(result);
    } catch (err) {
      setLoading('clustering', false);
      showError('clustering', err.message);
    }
  });

  $('btn-clu-reset').addEventListener('click', () => {
    $('form-clustering').reset();
    cluFields.forEach(k => {
      const el = document.getElementById(`err-${cluErrKeys[k]}`);
      if (el) el.textContent = '';
    });
    hideEl('result-clustering', 'error-clustering');
  });

  function renderClusteringResult(data) {
    $('clu-zone-value').textContent = data.cluster_label || `Zone ${data.assigned_cluster}`;
    $('clu-yield-value').textContent = data.cluster_mean_yield_kgha != null
      ? `${data.cluster_mean_yield_kgha.toLocaleString('en-IN', {maximumFractionDigits: 0})} kg/ha` : '—';
    $('clu-size-value').textContent = data.cluster_size != null
      ? `${data.cluster_size.toLocaleString('en-IN')} fields` : '—';
    $('clu-exploratory-note').textContent = data.exploratory_note || '';
    showEl('result-clustering');
  }

  // ── COMPLETE ANALYSIS ────────────────────────────────────────────────────────
  $('form-complete').addEventListener('submit', async (e) => {
    e.preventDefault();

    const raw = collectForm(e.target);

    // Build sub-payloads
    const regPayload = {
      Crop: raw['Crop'], Season: raw['Season'], State: raw['State'],
      Crop_Year: parseFloat(raw['Crop_Year']),
      Area: parseFloat(raw['Area']),
      Annual_Rainfall: parseFloat(raw['Annual_Rainfall']),
      Fertilizer: parseFloat(raw['Fertilizer']),
      Pesticide: parseFloat(raw['Pesticide']),
      Avg_Temperature: parseFloat(raw['Avg_Temperature']),
      Max_Temperature: parseFloat(raw['Max_Temperature']),
      Min_Temperature: parseFloat(raw['Min_Temperature']),
    };

    const clsPayload = {
      N: parseFloat(raw['N']), P: parseFloat(raw['P']), K: parseFloat(raw['K']),
      ph: parseFloat(raw['ph']),
      temperature: parseFloat(raw['cls_temperature']),
      humidity: parseFloat(raw['cls_humidity']),
      rainfall: parseFloat(raw['cls_rainfall']),
    };

    const cluPayload = {
      soil_ph: parseFloat(raw['clu_soil_ph']),
      organic_matter: parseFloat(raw['clu_organic_matter']),
      sand_pct: parseFloat(raw['clu_sand_pct']),
      ndvi: parseFloat(raw['clu_ndvi']),
      temperature: parseFloat(raw['clu_temperature']),
      humidity: parseFloat(raw['clu_humidity']),
      rainfall: parseFloat(raw['clu_rainfall']),
    };

    hideEl('complete-results', 'error-complete');
    showEl('loading-complete');

    try {
      const result = await Api.completeAnalysis({
        regression: regPayload,
        classification: clsPayload,
        clustering: cluPayload,
      });
      hideEl('loading-complete');
      renderCompleteResult(result);
    } catch (err) {
      hideEl('loading-complete');
      $('error-complete-msg').textContent = err.message;
      showEl('error-complete');
    }
  });

  $('btn-cp-reset').addEventListener('click', () => {
    $('form-complete').reset();
    hideEl('complete-results', 'error-complete', 'loading-complete');
  });

  function renderCompleteResult(data) {
    const results = data.results || {};

    // Regression card
    const reg = results.regression || {};
    $('cp-reg-status').textContent = reg.status === 'success' ? 'OK' : 'Error';
    $('cp-reg-status').className = `result-card-status ${reg.status === 'success' ? 'success' : 'error'}`;
    if (reg.status === 'success') {
      $('cp-reg-body').innerHTML = `
        <span class="card-value">${reg.predicted_yield != null ? reg.predicted_yield.toLocaleString('en-IN', {maximumFractionDigits: 1}) : '—'}</span>
        <span style="color:var(--ink-400);font-size:var(--text-sm)">kg per unit area</span>
      `;
    } else {
      $('cp-reg-body').textContent = reg.error || 'Missing or invalid regression inputs.';
    }

    // Classification card
    const cls = results.classification || {};
    $('cp-cls-status').textContent = cls.status === 'success' ? 'OK' : 'Error';
    $('cp-cls-status').className = `result-card-status ${cls.status === 'success' ? 'success' : 'error'}`;
    if (cls.status === 'success') {
      const top3 = (cls.top_3_candidates || []).map((c, i) =>
        `<span style="color:var(--ink-400);font-size:var(--text-xs)">${i+1}.</span> <strong>${c.crop}</strong> (${(c.probability*100).toFixed(1)}%)`
      ).join('<br>');
      $('cp-cls-body').innerHTML = `
        <span class="card-value" style="color:var(--accent);font-family:var(--font-body);text-transform:capitalize">${cls.recommended_crop || '—'}</span>
        <div style="margin-top:8px;line-height:1.9;font-size:var(--text-sm)">${top3}</div>
      `;
    } else {
      $('cp-cls-body').textContent = cls.error || 'Missing or invalid classification inputs.';
    }

    // Clustering card
    const clu = results.clustering || {};
    $('cp-clu-status').textContent = clu.status === 'success' ? 'OK' : 'Error';
    $('cp-clu-status').className = `result-card-status ${clu.status === 'success' ? 'success' : 'error'}`;
    if (clu.status === 'success') {
      $('cp-clu-body').innerHTML = `
        <span class="card-value">${clu.cluster_label || `Zone ${clu.assigned_cluster}`}</span>
        ${clu.cluster_mean_yield_kgha != null
          ? `<span style="color:var(--ink-400);font-size:var(--text-sm)">Zone mean yield: ${clu.cluster_mean_yield_kgha.toLocaleString('en-IN', {maximumFractionDigits: 0})} kg/ha</span>`
          : ''}
      `;
    } else {
      $('cp-clu-body').textContent = clu.error || 'Missing or invalid clustering inputs.';
    }

    showEl('complete-results');
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  checkHealth();
  loadMetadata();
});
