/**
 * validation.js — Client-side form validation helpers
 * Returns error strings keyed by field name, or empty object if valid.
 */

const Validation = (() => {

  function num(val, label, lo, hi) {
    const v = parseFloat(val);
    if (val === '' || val === null || val === undefined) return `${label} is required`;
    if (isNaN(v)) return `${label} must be a number`;
    if (lo !== undefined && v < lo) return `${label} must be ≥ ${lo}`;
    if (hi !== undefined && v > hi) return `${label} must be ≤ ${hi}`;
    return null;
  }

  function str(val, label) {
    if (!val || !val.toString().trim()) return `${label} is required`;
    return null;
  }

  function regression(data) {
    const errors = {};
    const e = (key, msg) => { if (msg) errors[key] = msg; };
    e('Crop',            str(data.Crop, 'Crop'));
    e('Season',          str(data.Season, 'Season'));
    e('State',           str(data.State, 'State'));
    e('Crop_Year',       num(data.Crop_Year, 'Crop Year', 1990, 2030));
    e('Area',            num(data.Area, 'Area', 0.1));
    e('Annual_Rainfall', num(data.Annual_Rainfall, 'Annual Rainfall', 0));
    e('Fertilizer',      num(data.Fertilizer, 'Fertilizer', 0));
    e('Pesticide',       num(data.Pesticide, 'Pesticide', 0));
    e('Avg_Temperature', num(data.Avg_Temperature, 'Avg Temperature', -20, 60));
    e('Max_Temperature', num(data.Max_Temperature, 'Max Temperature', -20, 70));
    e('Min_Temperature', num(data.Min_Temperature, 'Min Temperature', -30, 60));
    const maxT = parseFloat(data.Max_Temperature);
    const minT = parseFloat(data.Min_Temperature);
    if (!isNaN(maxT) && !isNaN(minT) && maxT <= minT) {
      errors['Max_Temperature'] = 'Max temperature must be greater than Min temperature';
    }
    return errors;
  }

  function classification(data) {
    const errors = {};
    const e = (key, msg) => { if (msg) errors[key] = msg; };
    e('N',           num(data.N, 'Nitrogen (N)', 0, 300));
    e('P',           num(data.P, 'Phosphorus (P)', 0, 300));
    e('K',           num(data.K, 'Potassium (K)', 0, 300));
    e('ph',          num(data.ph, 'Soil pH', 0, 14));
    e('temperature', num(data.temperature, 'Temperature', -10, 55));
    e('humidity',    num(data.humidity, 'Humidity', 0, 100));
    e('rainfall',    num(data.rainfall, 'Rainfall', 0));
    return errors;
  }

  function clustering(data) {
    const errors = {};
    const e = (key, msg) => { if (msg) errors[key] = msg; };
    e('soil_ph',        num(data.soil_ph, 'Soil pH', 3, 10));
    e('organic_matter', num(data.organic_matter, 'Organic Matter', 0, 20));
    e('sand_pct',       num(data.sand_pct, 'Sand Content', 0, 100));
    e('ndvi',           num(data.ndvi, 'NDVI', 0, 1));
    e('temperature',    num(data.temperature, 'Temperature', -10, 55));
    e('humidity',       num(data.humidity, 'Humidity', 0, 100));
    e('rainfall',       num(data.rainfall, 'Rainfall', 0));
    return errors;
  }

  function showErrors(errors, prefix) {
    let hasErrors = false;
    Object.keys(errors).forEach(key => {
      const el = document.getElementById(`err-${prefix}-${key}`);
      const inputEl = document.querySelector(`[name="${key}"]`);
      if (el) {
        el.textContent = errors[key] || '';
        hasErrors = true;
      }
      if (inputEl && errors[key]) {
        inputEl.classList.add('error');
      }
    });
    return hasErrors;
  }

  function clearErrors(prefix, fields) {
    fields.forEach(key => {
      const el = document.getElementById(`err-${prefix}-${key}`);
      if (el) el.textContent = '';
      const inputEl = document.querySelector(`#panel-${prefix} [name="${key}"]`) ||
                      document.querySelector(`[name="${key}"]`);
      if (inputEl) inputEl.classList.remove('error');
    });
  }

  return { regression, classification, clustering, showErrors, clearErrors };
})();

window.Validation = Validation;
