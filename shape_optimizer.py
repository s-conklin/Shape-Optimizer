import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import sys

# Import pluggable interface
sys.path.insert(0, os.path.dirname(__file__))
from stuff_model_interface import PitchScoutStuffModel
import synthetic_comps
import csv_import

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitch Scout — Shape Optimizer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #efece5;
    color: #3a352c;
}
.stApp { background: #efece5; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.04em; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

.stTabs [data-baseweb="tab-list"] { background: #efece5; border-bottom: 1px solid #e0dbd0; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif; font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; color: #8c857a;
    padding: 0.6rem 1.4rem; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #185FA5; border-bottom: 2px solid #185FA5; background: transparent; }

.stSelectbox > div > div {
    background: #faf8f4; border: 1px solid #d8d2c6; border-radius: 6px;
    font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #3a352c;
}
.stSelectbox input, .stSelectbox div[data-baseweb="select"] input {
    color: #3a352c !important; -webkit-text-fill-color: #3a352c !important;
    font-family: 'DM Mono', monospace;
}
.stSelectbox input::placeholder { color: #8c857a !important; -webkit-text-fill-color: #8c857a !important; }
div[data-baseweb="popover"] li, ul[role="listbox"] li {
    color: #3a352c; font-family: 'DM Mono', monospace; font-size: 0.8rem;
}
.stTextInput > div > div > input {
    background: #faf8f4; border: 1px solid #e0dbd0; border-radius: 6px;
    font-family: 'DM Mono', monospace; color: #3a352c;
}
.stButton > button {
    background: #185FA5; color: #fff;
    font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 0.9rem;
    letter-spacing: 0.1em; text-transform: uppercase; border: none;
    border-radius: 6px; padding: 0.5rem 1.5rem;
}
.stButton > button:hover { background: #0C447C; }
.stRadio > div { flex-direction: row; gap: 1rem; }
.stRadio label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #8c857a; }
div[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
DATA_DIR          = os.path.join(os.path.dirname(__file__), 'data')
VALID_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'FS', 'ST', 'SV']
FASTBALL_TYPES    = {'FF', 'SI', 'FC'}
VAA_PITCH_TYPES   = {'FF', 'SI'}

PITCH_NAMES = {
    'FF': '4-Seam FB', 'SI': 'Sinker',   'FC': 'Cutter',
    'SL': 'Slider',    'ST': 'Sweeper',  'CU': 'Curveball',
    'KC': 'Knuckle-CU','CH': 'Changeup', 'FS': 'Splitter', 'SV': 'Screwball',
}
PITCH_COLORS = {
    'FF': '#FF6B6B', 'SI': '#FF9F43', 'FC': '#FFC312',
    'SL': '#54A0FF', 'ST': '#5F27CD', 'CU': '#00D2D3',
    'KC': '#01ABC3', 'CH': '#10AC84', 'FS': '#EE5A24', 'SV': '#C8D6E5',
}

def pitch_name(pt): return PITCH_NAMES.get(pt, pt)
def pitch_color(pt): return PITCH_COLORS.get(pt, '#8892b0')


def to_first_last(name):
    """Convert a stored 'Last, First' name to 'First Last' for display.
    Leaves names without a comma unchanged (already 'First Last' or single token).
    Handles suffixes that follow the first name after the comma, e.g.
    'Tatis Jr., Fernando' -> 'Fernando Tatis Jr.'."""
    if not name or ',' not in name:
        return name or ''
    last, first = name.split(',', 1)
    last, first = last.strip(), first.strip()
    if not first:
        return last
    return f"{first} {last}"

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    import xgboost as xgb
    data = {}

    # Load non-model pkl files
    for fname, key in [
        ('norm_tables.pkl',            'norm_tables'),
        ('pitcher_stuff_profiles.pkl', 'profiles'),
        ('feature_importance.pkl',     'feature_importance'),
        ('pitcher_comp_profiles.pkl',  'comp_profiles'),
        ('sensitivity_radii.pkl',      'sensitivity_radii'),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data[key] = pickle.load(f)
        else:
            data[key] = None

    # Load stuff models using native XGBoost format (avoids version mismatch)
    meta_path = os.path.join(DATA_DIR, 'stuff_models_meta.pkl')
    native_dir = os.path.join(DATA_DIR, 'stuff_models_native')

    if os.path.exists(meta_path) and os.path.exists(native_dir):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stuff_models = {}
        for key, entry in meta.items():
            mpt = {'feature_cols': entry['feature_cols']}
            for metric in ['rv', 'hard_hit', 'xwoba']:
                if metric in entry:
                    model_path = entry[metric]
                    # Path stored in meta may reference Colab — remap to local
                    fname_only = os.path.basename(model_path)
                    local_path = os.path.join(native_dir, fname_only)
                    if os.path.exists(local_path):
                        m = xgb.Booster()
                        m.load_model(local_path)
                        mpt[metric] = m
            stuff_models[key] = mpt
        data['stuff_models'] = stuff_models
    else:
        # Fallback to pickle (may show version warning)
        pkl_path = os.path.join(DATA_DIR, 'stuff_models.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data['stuff_models'] = pickle.load(f)
        else:
            data['stuff_models'] = None

    return data

# ── Grade helpers (v3 — RV-based) ────────────────────────────────────────
# Fixed scale factor: ~1 SD of actual pitch-level RV across the league.
# Keeps Stuff+ on an interpretable 100-scale where 15 pts ≈ 1 SD above avg.
STUFF_SCALE = 0.15

def compute_stuff_plus(pred_rv, p_throws, stand, pt, norm_tables):
    """Stuff+ within pitch type. 100 = league avg, higher = better."""
    key     = (p_throws, stand, pt)
    lg_mean = norm_tables['rv_mean'].get(key)
    correction = norm_tables.get('stuff_plus_correction', 0.0)
    if lg_mean is None:
        return 100.0
    return round(100 + 100 * (lg_mean - pred_rv) / STUFF_SCALE - correction, 1)

def compute_arsenal_plus(pred_rv, pt, norm_tables):
    """Arsenal+ cross-type. 100 = avg across all pitch types."""
    lg_mean = norm_tables['rv_mean_all_by_pt'].get(pt)
    correction = norm_tables.get('stuff_plus_correction', 0.0)
    if lg_mean is None:
        return 100.0
    return round(100 + 100 * (lg_mean - pred_rv) / STUFF_SCALE - correction, 1)

def compute_contact_plus(pred_hh, pred_xw, p_throws, stand, pt, norm_tables):
    """Contact+. Higher = better contact suppression."""
    key   = (p_throws, stand, pt)
    lg_hh = norm_tables['hh_mean'].get(key)
    lg_xw = norm_tables['xwoba_mean'].get(key)
    if lg_hh and pred_hh > 0 and lg_xw and pred_xw > 0:
        return round(((100 * lg_hh / pred_hh) + (100 * lg_xw / pred_xw)) / 2, 1)
    return 100.0

def grade_color(val, base=100):
    if val >= base + 15: return '#1D9E75'
    if val >= base + 5:  return '#378ADD'
    if val >= base - 5:  return '#3a352c'
    if val >= base - 15: return '#f59e0b'
    return '#ef4444'

# ── Prediction helper (uses pluggable interface) ──────────────────────────
def build_features(pt, pfx_x, pfx_z, spin_eff, velo, profile_info):
    """Build feature dict for a given shape."""
    fixed    = profile_info['fixed']
    semi     = profile_info['semi_fixed']
    p_throws = profile_info.get('p_throws', 'R')
    fb_velo  = profile_info.get('primary_fb_velo')
    fb_pfx_x = profile_info.get('primary_fb_pfx_x')
    fb_pfx_z = profile_info.get('primary_fb_pfx_z')

    if pt in FASTBALL_TYPES or fb_velo is None:
        vdiff, xdiff, zdiff = 0.0, 0.0, 0.0
    else:
        vdiff = (fb_velo - velo)
        xdiff = (pfx_x - (fb_pfx_x or 0.0))
        zdiff = (pfx_z - (fb_pfx_z or 0.0))

    ext      = fixed.get('release_extension', 6.0) or 6.0
    spin_ax  = semi.get('spin_axis_mean') or 180.0
    ssw_val  = np.radians(spin_ax) * (spin_eff or 0.0)

    # VAA: derive from the CANDIDATE shape rather than using a fixed measured value.
    # VAA is a consequence of vertical movement + release geometry, so when the
    # optimizer changes pfx_z the approach angle must change too. We use the delta
    # method: measured current VAA + (shape_vaa(candidate) - shape_vaa(current)),
    # which keeps the value on the measured scale the model trained on.
    measured_vaa = fixed.get('vaa_mean')
    cur_pfx_x = profile_info.get('optimizable', {}).get('pfx_x_mean')
    cur_pfx_z = profile_info.get('optimizable', {}).get('pfx_z_mean')
    vaa_val = measured_vaa if measured_vaa is not None else -5.0
    if measured_vaa is not None and cur_pfx_x is not None and cur_pfx_z is not None:
        try:
            from tunneling import optimized_vaa
            rpy = fixed.get('release_pos_y') or (60.5 - ext)
            derived = optimized_vaa(
                measured_vaa, cur_pfx_x, cur_pfx_z, pfx_x, pfx_z, velo,
                fixed.get('release_pos_x', 0.0) or 0.0, rpy,
                fixed.get('release_pos_z', 6.0) or 6.0,
                vx0_actual=fixed.get('vx0'), vy0_actual=fixed.get('vy0'),
                vz0_actual=fixed.get('vz0'), ay_actual=fixed.get('ay'),
            )
            if derived is not None:
                vaa_val = derived
        except Exception:
            pass

    return {
        'release_speed':       velo,
        'pfx_x_in':            pfx_x,
        'pfx_z_in':            pfx_z,
        'release_pos_x':       fixed.get('release_pos_x', 0.0) or 0.0,
        'release_pos_z':       fixed.get('release_pos_z', 6.0) or 6.0,
        'release_extension':   ext,
        'velo_diff_fb':        vdiff,
        'pfx_x_diff_fb':       xdiff,
        'pfx_z_diff_fb':       zdiff,
        'vaa':                 vaa_val,
        'spin_axis':           spin_ax,
        'spin_efficiency_raw': spin_eff or 0.0,
        'ssw_interaction':     ssw_val,
    }

def predict_grades(pt, pfx_x, pfx_z, spin_eff, velo, profile_info,
                   norm_tables, stuff_models, stand='R'):
    """Predict grades using the pluggable interface."""
    p_throws = profile_info.get('p_throws', 'R')
    model    = PitchScoutStuffModel(stuff_models, norm_tables, p_throws, stand, pt)
    features = build_features(pt, pfx_x, pfx_z, spin_eff, velo, profile_info)
    return model.predict_grades_full(features)

def predict_weighted_grades(pt, pfx_x, pfx_z, spin_eff, velo, profile_info,
                             norm_tables, stuff_models):
    """Weighted grades across both handedness matchups."""
    rhh_w = norm_tables.get('rhh_weight', 0.58)
    lhh_w = norm_tables.get('lhh_weight', 0.42)
    gr    = predict_grades(pt, pfx_x, pfx_z, spin_eff, velo, profile_info,
                           norm_tables, stuff_models, stand='R')
    gl    = predict_grades(pt, pfx_x, pfx_z, spin_eff, velo, profile_info,
                           norm_tables, stuff_models, stand='L')
    combined = {}
    for metric in ['stuff_plus', 'arsenal_plus', 'contact_plus', 'pred_rv']:
        vr = gr.get(metric)
        vl = gl.get(metric)
        if vr is not None and vl is not None:
            combined[metric] = round(vr * rhh_w + vl * lhh_w, 1)
        elif vr is not None:
            combined[metric] = vr
        elif vl is not None:
            combined[metric] = vl
    return combined, gr, gl

# ── Optimizer: coarse grid + scipy refinement ────────────────────────────
def run_optimizer(pt, profile_info, norm_tables, stuff_models, stand='R'):
    """
    Two-phase optimizer:
    1. Coarse 20x20 grid over pfx_x x pfx_z to find global region
    2. Scipy L-BFGS-B refinement starting from best grid point
    Spin efficiency fixed at pitcher's current value — it constrains
    what movement is achievable (via comp filtering) but is not itself
    a recommendation target.
    Returns (opt_pfx_x, opt_pfx_z, curr_spin_eff, best_grades)
    """
    p_throws  = profile_info.get('p_throws', 'R')
    opt       = profile_info['optimizable']
    semi      = profile_info['semi_fixed']
    velo      = semi['velo_mean']
    curr_se   = profile_info['fixed'].get('spin_efficiency')

    pfx_x_bounds = (opt['pfx_x_lo'], opt['pfx_x_hi'])
    pfx_z_bounds = (opt['pfx_z_lo'], opt['pfx_z_hi'])
    bounds       = [pfx_x_bounds, pfx_z_bounds]

    model = PitchScoutStuffModel(stuff_models, norm_tables, p_throws, stand, pt)

    def objective(x):
        pfx_x, pfx_z = x
        features = build_features(pt, pfx_x, pfx_z, curr_se, velo, profile_info)
        return model.predict_rv(features)

    # Phase 1 — coarse 20x20 grid
    x_vals = np.linspace(pfx_x_bounds[0], pfx_x_bounds[1], 20)
    z_vals = np.linspace(pfx_z_bounds[0], pfx_z_bounds[1], 20)
    best_rv    = float('inf')
    best_x0    = [opt.get('pfx_x_mean', (pfx_x_bounds[0]+pfx_x_bounds[1])/2),
                  opt.get('pfx_z_mean', (pfx_z_bounds[0]+pfx_z_bounds[1])/2)]

    for px in x_vals:
        for pz in z_vals:
            rv = objective([px, pz])
            if rv < best_rv:
                best_rv = rv
                best_x0 = [px, pz]

    # Phase 2 — scipy refinement from best grid point
    try:
        result = minimize(
            objective, x0=best_x0, bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-9},
        )
        opt_pfx_x, opt_pfx_z = result.x
    except Exception:
        opt_pfx_x, opt_pfx_z = best_x0

    best_grades = predict_grades(
        pt, float(opt_pfx_x), float(opt_pfx_z), curr_se,
        velo, profile_info, norm_tables, stuff_models, stand
    )

    return float(opt_pfx_x), float(opt_pfx_z), curr_se, best_grades

# ── Shape card renderer ───────────────────────────────────────────────────
def render_shape_card(title, pt, current_grades, opt_grades,
                      current_pfx_x, current_pfx_z,
                      opt_pfx_x, opt_pfx_z, velo, color,
                      current_vaa=None, opt_vaa=None):

    sp_curr = current_grades.get('stuff_plus', 100)
    sp_opt  = opt_grades.get('stuff_plus', 100)
    ap_curr = current_grades.get('arsenal_plus', 100)
    ap_opt  = opt_grades.get('arsenal_plus', 100)
    cp_curr = current_grades.get('contact_plus', 100)
    cp_opt  = opt_grades.get('contact_plus', 100)
    rv_curr = current_grades.get('pred_rv', 0)
    rv_opt  = opt_grades.get('pred_rv', 0)

    delta_sp = round(sp_opt - sp_curr, 1)
    delta_ap = round(ap_opt - ap_curr, 1)
    delta_cp = round(cp_opt - cp_curr, 1)
    delta_rv = round((rv_opt - rv_curr) * 100, 2)  # RV/100

    def delta_html(d, invert=False):
        better = (d < 0) if invert else (d > 0)
        col  = '#1D9E75' if better else ('#ef4444' if abs(d) > 0.5 else '#8c857a')
        sign = '+' if d > 0 else ''
        return f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:{col};">{sign}{d}</span>'

    def grade_pill(val, label, large=False):
        c = grade_color(val)
        size = '2.2rem' if large else '1.4rem'
        return (
            f'<div style="text-align:center;">'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:{size};'
            f'font-weight:900;color:{c};line-height:1;">{val:.0f}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.55rem;'
            f'color:#8c857a;text-transform:uppercase;letter-spacing:0.1em;">{label}</div>'
            f'</div>'
        )

    def arrow_stat(label, curr_val, opt_val, unit='&quot;'):
        delta = round(opt_val - curr_val, 1)
        sign  = '+' if delta > 0 else ''
        dcol  = '#1D9E75' if delta > 0 else ('#ef4444' if delta < -0.05 else '#8c857a')
        return (
            f'<div style="background:#efece5;border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:0.6rem;font-weight:700;'
            f'color:#8c857a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">{label}</div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<span style="font-family:DM Mono,monospace;font-size:0.85rem;color:#8c857a;">{curr_val:.1f}{unit}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#8c857a;">→</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:1.1rem;font-weight:500;color:#3a352c;">{opt_val:.1f}{unit}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.85rem;color:{dcol};">{sign}{delta:.1f}{unit}</span>'
            f'</div></div>'
        )

    rv_sign = '+' if delta_rv > 0 else ''
    rv_col  = '#ef4444' if delta_rv > 0 else '#1D9E75'

    st.markdown(
        f'<div style="background:#faf8f4;border:1px solid rgba(255,255,255,0.07);'
        f'border-top:3px solid {color};border-radius:0 0 8px 8px;padding:1.1rem 1.2rem;">'
        f'<div style="font-family:Barlow Condensed,sans-serif;font-size:0.6rem;font-weight:700;'
        f'color:{color};letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.9rem;">{title}</div>'

        # Grade row — original pill layout
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:1rem;padding-bottom:0.9rem;border-bottom:1px solid rgba(255,255,255,0.06);">'
        f'<div>'
        f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#8c857a;margin-bottom:4px;">CURRENT</div>'
        f'<div style="display:flex;gap:1.5rem;align-items:flex-end;">'
        f'{grade_pill(sp_curr, "Stuff+", large=True)}'
        f'{grade_pill(ap_curr, "Arsenal+")}'
        f'{grade_pill(cp_curr, "Contact+")}'
        f'</div>'
        f'</div>'
        f'<div style="font-size:1.2rem;color:#8c857a;">→</div>'
        f'<div>'
        f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#378ADD;margin-bottom:4px;">OPTIMIZED</div>'
        f'<div style="display:flex;gap:1.5rem;align-items:flex-end;">'
        f'{grade_pill(sp_opt, "Stuff+", large=True)}'
        f'{grade_pill(ap_opt, "Arsenal+")}'
        f'{grade_pill(cp_opt, "Contact+")}'
        f'</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#8c857a;margin-bottom:6px;">DELTA</div>'
        f'<div style="display:flex;flex-direction:column;gap:4px;align-items:flex-end;">'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">Stuff+</span>'
        f'{delta_html(delta_sp)}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">Arsenal+</span>'
        f'{delta_html(delta_ap)}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">Contact+</span>'
        f'{delta_html(delta_cp)}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin-top:2px;padding-top:4px;'
        f'border-top:1px solid rgba(255,255,255,0.06);">'
        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">RV/100</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:{rv_col};">'
        f'{rv_sign}{delta_rv}</span></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # Shape stats — current → target  delta format
    shape_stats = (
        arrow_stat('Horizontal Break', current_pfx_x, opt_pfx_x) +
        arrow_stat('Vertical Break',   current_pfx_z, opt_pfx_z) +
        arrow_stat('Velocity',         velo, velo, unit=' mph')
    )
    # VAA shown for fastballs/sinkers, where approach angle is a meaningful lever
    # and is part of the stuff model's feature set. It's a derived consequence of
    # the shape, so it changes as the optimizer moves vertical break.
    if pt in VAA_PITCH_TYPES and current_vaa is not None and opt_vaa is not None:
        shape_stats += arrow_stat('Approach Angle (VAA)', current_vaa, opt_vaa, unit='\u00b0')
    st.markdown(shape_stats, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────


# ── Polar movement chart (Savant style) ──────────────────────────────────
def render_movement_chart(pt, current_pfx_x, current_pfx_z,
                           opt_pfx_x, opt_pfx_z, comp_bounds, color, pitcher_name,
                           arsenal_pitches=None, arsenal_tunnel_data=None):
    """
    arsenal_pitches: list of dicts {pt, name, pfx_x, pfx_z, color, is_pair}
      Other pitches in pitcher's arsenal (current shapes only)
    arsenal_tunnel_data: dict {pt: {curr_lbr, opt_lbr, curr_tun, opt_tun, curr_plate, opt_plate}}
      Tunneling metrics for each arsenal pitch vs the selected pitch
    """
    import streamlit.components.v1 as components
    import json
    cx_f = round(float(current_pfx_x), 1)
    cz_f = round(float(current_pfx_z), 1)
    ox_f = round(float(opt_pfx_x), 1)
    oz_f = round(float(opt_pfx_z), 1)
    dx = round(ox_f - cx_f, 1)
    dz = round(oz_f - cz_f, 1)
    sign = lambda v: ('+' if v > 0 else '') + str(v)
    dx_s = sign(dx); dz_s = sign(dz)
    dx_col = '#1D9E75' if dx != 0 else '#888'
    dz_col = '#1D9E75' if dz != 0 else '#888'

    # Real comp pitcher data
    comp_pitchers = comp_bounds.get('comp_pitchers', []) if comp_bounds else []
    comp_json = json.dumps(comp_pitchers)

    # Arsenal data for the arsenal toggle
    arsenal_json = json.dumps(arsenal_pitches or [])
    tunnel_json  = json.dumps(arsenal_tunnel_data or {})

    pn = PITCH_NAMES.get(pt, pt)
    CUR_COLOR = '#3b82f6'
    TGT_COLOR = '#f59e0b'

    html_parts = []
    html_parts.append('<div style="background:#faf8f4;border:1px solid #e0dbd0;border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:0.75rem;font-family:sans-serif;width:100%;">')
    html_parts.append('<div style="margin-bottom:10px;">')
    html_parts.append(f'<div><span style="font-size:14px;font-weight:700;color:#3a352c;">{pn} \u2014 Movement Profile (Induced Break)</span>')
    html_parts.append(f'<span style="font-size:12px;color:#888;margin-left:10px;">{pitcher_name}</span></div>')
    html_parts.append('<div style="display:flex;gap:18px;align-items:center;margin-top:8px;flex-wrap:wrap;">')
    # Arsenal toggle
    html_parts.append(f'<div onclick="window[\'ta_{pt}\']()" style="display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;">')
    html_parts.append(f'<div id="toga_{pt}" style="width:32px;height:18px;border-radius:9px;background:#ccc;position:relative;transition:background .18s;flex-shrink:0;">')
    html_parts.append(f'<div id="togaball_{pt}" style="position:absolute;width:12px;height:12px;border-radius:50%;background:white;top:3px;left:3px;transition:left .18s;box-shadow:0 1px 3px rgba(0,0,0,0.2);"></div></div>')
    html_parts.append('<span style="font-size:12px;color:#888;white-space:nowrap;">show arsenal</span></div>')
    # Comps toggle
    html_parts.append(f'<div onclick="window[\'tc_{pt}\']()" style="display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;">')
    html_parts.append(f'<div id="tog_{pt}" style="width:32px;height:18px;border-radius:9px;background:#ccc;position:relative;transition:background .18s;flex-shrink:0;">')
    html_parts.append(f'<div id="togball_{pt}" style="position:absolute;width:12px;height:12px;border-radius:50%;background:white;top:3px;left:3px;transition:left .18s;box-shadow:0 1px 3px rgba(0,0,0,0.2);"></div></div>')
    html_parts.append('<span style="font-size:12px;color:#888;white-space:nowrap;">show comps</span></div>')
    html_parts.append('</div></div>')
    html_parts.append(f'<div style="position:relative;width:100%;height:600px;">')
    html_parts.append(f'<canvas id="mc_{pt}" role="img" style="position:absolute;top:0;left:0;width:100%;height:100%;" aria-label="Movement profile for {pn}">Current: {cx_f} H {cz_f} V. Target: {ox_f} H {oz_f} V.</canvas>')
    # Tooltip overlay
    html_parts.append(f'<div id="tt_{pt}" style="position:absolute;display:none;background:rgba(0,0,0,0.82);color:white;font-size:11px;padding:5px 9px;border-radius:5px;pointer-events:none;white-space:nowrap;z-index:10;"></div>')
    html_parts.append('</div>')
    html_parts.append('<div style="display:flex;gap:20px;margin-top:10px;flex-wrap:wrap;align-items:center;">')
    html_parts.append(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;"><svg width="12" height="12"><circle cx="6" cy="6" r="6" fill="{CUR_COLOR}"/></svg> current shape</div>')
    html_parts.append(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;"><svg width="14" height="12"><polygon points="7,0 14,12 0,12" fill="{TGT_COLOR}"/></svg> optimal target</div>')
    html_parts.append(f'<div id="cleg_{pt}" style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;opacity:0.35;transition:opacity .18s;">')
    html_parts.append('<svg width="12" height="12"><circle cx="6" cy="6" r="6" fill="#94a3b8"/></svg> comp group &nbsp;')
    html_parts.append('<svg width="12" height="12"><circle cx="6" cy="6" r="6" fill="#f97316"/></svg> closest to optimal (hover for name)')
    html_parts.append('</div></div>')
    html_parts.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;">')
    html_parts.append(f'<div style="background:#faf8f4;border:1px solid #e0dbd0;border-radius:6px;padding:8px 14px;">'
                      f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px;">Horizontal Break</div>'
                      f'<div style="display:flex;align-items:center;gap:8px;">'
                      f'<span style="font-size:16px;font-weight:600;color:#1e3a8a;">{cx_f}&quot;</span>'
                      f'<span style="font-size:13px;color:#a89f92;">\u2192</span>'
                      f'<span style="font-size:16px;font-weight:700;color:#92400e;">{ox_f}&quot;</span>'
                      f'<span style="font-size:12px;font-weight:500;color:{dx_col};margin-left:2px;">{dx_s}&quot;</span>'
                      f'</div></div>')
    html_parts.append(f'<div style="background:#faf8f4;border:1px solid #e0dbd0;border-radius:6px;padding:8px 14px;">'
                      f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px;">Vertical Break</div>'
                      f'<div style="display:flex;align-items:center;gap:8px;">'
                      f'<span style="font-size:16px;font-weight:600;color:#1e3a8a;">{cz_f}&quot;</span>'
                      f'<span style="font-size:13px;color:#a89f92;">\u2192</span>'
                      f'<span style="font-size:16px;font-weight:700;color:#92400e;">{oz_f}&quot;</span>'
                      f'<span style="font-size:12px;font-weight:500;color:{dz_col};margin-left:2px;">{dz_s}&quot;</span>'
                      f'</div></div>')
    html_parts.append('</div></div>')
    html = ''.join(html_parts)

    js = """
<script>
(function(){
  var showC=false;
  var showA=false;
  var canv=document.getElementById('mc_PT');
  var ctx=canv.getContext('2d');
  var tt=document.getElementById('tt_PT');
  var W,H,CX,CY,SC,DPR;
  var cur={x:CX_F,y:CZ_F};
  var opt={x:OX_F,y:OZ_F};
  var allComps=COMP_JSON;
  var arsenal=ARSENAL_JSON;
  var tunnelData=TUNNEL_JSON;

  window['tc_PT']=function(){
    showC=!showC;
    document.getElementById('tog_PT').style.background=showC?'#1D9E75':'#ccc';
    document.getElementById('togball_PT').style.left=showC?'16px':'3px';
    document.getElementById('cleg_PT').style.opacity=showC?'1':'0.35';
    draw();
  };

  window['ta_PT']=function(){
    showA=!showA;
    document.getElementById('toga_PT').style.background=showA?'#1D9E75':'#ccc';
    document.getElementById('togaball_PT').style.left=showA?'16px':'3px';
    draw();
  };

  function setup(){
    DPR=window.devicePixelRatio||1;
    W=canv.offsetWidth; H=canv.offsetHeight;
    canv.width=W*DPR; canv.height=H*DPR;
    ctx.setTransform(DPR,0,0,DPR,0,0);
    CX=W/2; CY=H/2;
    SC=Math.min(W,H)*0.43/24;
  }
  function xy(h,v){return {x:CX-h*SC,y:CY-v*SC};}

  // Identify 3 comps closest to optimal target
  function getClosest(){
    var dists=allComps.map(function(c,i){
      var dx=c.pfx_x-opt.x, dz=c.pfx_z-opt.y;
      return {i:i, d:Math.sqrt(dx*dx+dz*dz)};
    });
    dists.sort(function(a,b){return a.d-b.d;});
    var closest={};
    for(var k=0;k<Math.min(3,dists.length);k++) closest[dists[k].i]=true;
    return closest;
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    var maxR=24*SC;
    [24,18,12,6].forEach(function(r,i){
      ctx.beginPath(); ctx.arc(CX,CY,r*SC,0,Math.PI*2);
      ctx.fillStyle=i%2===0?'#e8f4fc':'#d6ecf7'; ctx.fill();
    });
    [6,12,18,24].forEach(function(r){
      ctx.beginPath(); ctx.arc(CX,CY,r*SC,0,Math.PI*2);
      ctx.strokeStyle='rgba(100,160,210,0.4)'; ctx.lineWidth=r===24?1.5:0.75; ctx.stroke();
    });
    ctx.beginPath(); ctx.moveTo(CX-maxR-2,CY); ctx.lineTo(CX+maxR+2,CY);
    ctx.moveTo(CX,CY-maxR-2); ctx.lineTo(CX,CY+maxR+2);
    ctx.strokeStyle='rgba(100,160,210,0.5)'; ctx.lineWidth=0.75; ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='bold 10px sans-serif';
    [6,12,18,24].forEach(function(r){
      ctx.textAlign='center'; ctx.textBaseline='top';
      ctx.fillText(r+'"',CX+r*SC,CY+5); ctx.fillText(r+'"',CX-r*SC,CY+5);
      if(r<24){ctx.textAlign='right'; ctx.textBaseline='middle'; ctx.fillText(r+'"',CX-5,CY-r*SC); ctx.fillText('-'+r+'"',CX-5,CY+r*SC);}
    });
    ctx.font='bold 11px sans-serif'; ctx.fillStyle='#333';
    ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.fillText('MORE RISE \u25b2',CX,CY-maxR-8);
    ctx.textBaseline='top'; ctx.fillText('\u25bc MORE DROP',CX,CY+maxR+8);
    ctx.textBaseline='middle';
    ctx.textAlign='left';  ctx.fillText('\u25ba 3B',CX+maxR+6,CY);
    ctx.textAlign='right'; ctx.fillText('1B \u25c4',CX-maxR-6,CY);

    // Comp cloud — real pitcher dots
    if(showC && allComps.length>0){
      var closest=getClosest();
      // Draw regular comps first
      allComps.forEach(function(c,i){
        if(closest[i]) return;
        var p=xy(c.pfx_x,c.pfx_z);
        ctx.beginPath(); ctx.arc(p.x,p.y,5,0,Math.PI*2);
        ctx.fillStyle='rgba(148,163,184,0.55)'; ctx.fill();
        ctx.strokeStyle='rgba(100,116,139,0.7)'; ctx.lineWidth=0.5; ctx.stroke();
      });
      // Draw closest 3 on top — orange/gold
      allComps.forEach(function(c,i){
        if(!closest[i]) return;
        var p=xy(c.pfx_x,c.pfx_z);
        ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2);
        ctx.fillStyle='rgba(249,115,22,0.85)'; ctx.fill();
        ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();
      });
    }

    // Arsenal pitches — other pitches in pitcher's arsenal
    if(showA && arsenal.length>0){
      arsenal.forEach(function(a){
        var p=xy(a.pfx_x, a.pfx_z);
        // Outer halo for paired pitches (those that tunnel with selected pitch)
        if(a.is_pair){
          ctx.beginPath(); ctx.arc(p.x, p.y, 16, 0, Math.PI*2);
          ctx.fillStyle='rgba(16,185,129,0.12)'; ctx.fill();
        }
        // White ring then colored dot
        ctx.beginPath(); ctx.arc(p.x, p.y, 11, 0, Math.PI*2);
        ctx.fillStyle='white'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI*2);
        ctx.fillStyle=a.color; ctx.fill();
        // Pitch type abbreviation
        ctx.fillStyle='white'; ctx.font='bold 9px sans-serif';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(a.pt, p.x, p.y);
      });
    }

    // Arrow
    var C=xy(cur.x,cur.y), O=xy(opt.x,opt.y);
    var ang=Math.atan2(O.y-C.y,O.x-C.x), off=13;
    ctx.beginPath(); ctx.moveTo(C.x+Math.cos(ang)*11,C.y+Math.sin(ang)*11);
    ctx.lineTo(O.x-Math.cos(ang)*off,O.y-Math.sin(ang)*off);
    ctx.strokeStyle='rgba(0,0,0,0.22)'; ctx.lineWidth=1.5;
    ctx.setLineDash([6,5]); ctx.stroke(); ctx.setLineDash([]);
    var tip={x:O.x-Math.cos(ang)*2,y:O.y-Math.sin(ang)*2};
    ctx.beginPath();
    ctx.moveTo(tip.x-Math.cos(ang-0.42)*off,tip.y-Math.sin(ang-0.42)*off);
    ctx.lineTo(tip.x,tip.y); ctx.lineTo(tip.x-Math.cos(ang+0.42)*off,tip.y-Math.sin(ang+0.42)*off);
    ctx.strokeStyle='rgba(0,0,0,0.28)'; ctx.lineWidth=1.5; ctx.stroke();

    // Labels before icons
    ctx.font='bold 10px sans-serif';
    var ICON_R=11, PAD=7, PH=17, PP=5;
    var cLbl=cur.x.toFixed(1)+'" / '+cur.y.toFixed(1)+'"';
    var oLbl=opt.x.toFixed(1)+'" / '+opt.y.toFixed(1)+'"';
    var cTW=ctx.measureText(cLbl).width;
    var oTW=ctx.measureText(oLbl).width;
    var dX=O.x-C.x, dY=O.y-C.y;
    var absDX=Math.abs(dX), absDY=Math.abs(dY);
    var cPrefY=0,oPrefY=0,cPrefX=0,oPrefX=0;
    if(absDY>=absDX){cPrefY=dY>0?-1:1; oPrefY=dY>0?1:-1;}
    else{cPrefX=dX>0?-1:1; oPrefX=dX>0?1:-1;}
    function genCands(dot,px,py){
      var r=ICON_R+PAD; var cands=[];
      if(py!=0&&px==0){cands.push({x:dot.x,y:dot.y+py*r,al:'center'});cands.push({x:dot.x+r,y:dot.y+py*r,al:'left'});cands.push({x:dot.x-r,y:dot.y+py*r,al:'right'});cands.push({x:dot.x,y:dot.y-py*r,al:'center'});}
      else if(px!=0&&py==0){var s=px>0?'left':'right';cands.push({x:dot.x+px*(r+2),y:dot.y,al:s});cands.push({x:dot.x+px*(r+2),y:dot.y-r,al:s});cands.push({x:dot.x+px*(r+2),y:dot.y+r,al:s});cands.push({x:dot.x-px*(r+2),y:dot.y,al:px<0?'left':'right'});}
      else{cands.push({x:dot.x,y:dot.y-r,al:'center'});cands.push({x:dot.x,y:dot.y+r,al:'center'});cands.push({x:dot.x+r,y:dot.y,al:'left'});cands.push({x:dot.x-r,y:dot.y,al:'right'});}
      return cands;
    }
    function bbox(pos,tw){var bx=pos.al==='center'?pos.x-tw/2-PP:pos.al==='left'?pos.x-PP:pos.x-tw-PP;return {x:bx,y:pos.y-PH/2,w:tw+PP*2,h:PH};}
    function rOver(a,b){return a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;}
    function cOver(cx,cy,cr,r){var nx=Math.max(r.x,Math.min(cx,r.x+r.w)),ny=Math.max(r.y,Math.min(cy,r.y+r.h));return (cx-nx)*(cx-nx)+(cy-ny)*(cy-ny)<cr*cr;}
    var cCands=genCands(C,cPrefX,cPrefY); var oCands=genCands(O,oPrefX,oPrefY);
    var cPos=cCands[0];
    for(var ci=0;ci<cCands.length;ci++){var b=bbox(cCands[ci],cTW);if(!cOver(O.x,O.y,ICON_R+2,b)){cPos=cCands[ci];break;}}
    var cBF=bbox(cPos,cTW); var oPos=oCands[0];
    for(var oi=0;oi<oCands.length;oi++){var b=bbox(oCands[oi],oTW);if(!cOver(C.x,C.y,ICON_R+2,b)&&!rOver(b,cBF)){oPos=oCands[oi];break;}}
    function pill(txt,pos,tw,fg,bc){
      ctx.textBaseline='middle';var b=bbox(pos,tw);
      ctx.fillStyle='rgba(255,255,255,0.97)';ctx.strokeStyle=bc;ctx.lineWidth=1.2;
      ctx.beginPath();ctx.rect(b.x,b.y,b.w,b.h);ctx.fill();ctx.stroke();
      ctx.fillStyle=fg;ctx.textAlign=pos.al;ctx.fillText(txt,pos.x,pos.y);
    }
    pill(cLbl,cPos,cTW,'#1d4ed8','rgba(59,130,246,0.5)');
    pill(oLbl,oPos,oTW,'#b45309','rgba(245,158,11,0.5)');

    // Icons on top
    ctx.beginPath(); ctx.arc(C.x,C.y,11,0,Math.PI*2); ctx.fillStyle='white'; ctx.fill();
    ctx.beginPath(); ctx.arc(C.x,C.y,9,0,Math.PI*2); ctx.fillStyle='#3b82f6'; ctx.fill();
    ctx.fillStyle='white'; ctx.font='bold 9px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText('C',C.x,C.y);
    var ts=11;
    ctx.beginPath(); ctx.arc(O.x,O.y,ts+5,0,Math.PI*2); ctx.fillStyle='rgba(245,158,11,0.15)'; ctx.fill();
    ctx.beginPath(); ctx.moveTo(O.x,O.y-ts); ctx.lineTo(O.x+ts*0.87,O.y+ts*0.5); ctx.lineTo(O.x-ts*0.87,O.y+ts*0.5);
    ctx.closePath(); ctx.fillStyle='#f59e0b'; ctx.fill(); ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();
    ctx.fillStyle='white'; ctx.font='bold 9px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText('T',O.x,O.y+1);
  }

  // Hover tooltip for comp dots and arsenal dots
  canv.addEventListener('mousemove',function(e){
    var rect=canv.getBoundingClientRect();
    var mx=(e.clientX-rect.left)*(canv.width/rect.width)/DPR;
    var my=(e.clientY-rect.top)*(canv.height/rect.height)/DPR;
    var hit=null;
    var hitType=null;

    // Check arsenal dots first (rendered later, on top)
    if(showA && arsenal.length>0){
      arsenal.forEach(function(a){
        var p=xy(a.pfx_x,a.pfx_z);
        var dx=mx-p.x, dy=my-p.y;
        if(dx*dx+dy*dy<=11*11){hit=a; hitType='arsenal';}
      });
    }

    // Then check comp dots
    if(!hit && showC && allComps.length>0){
      var closest=getClosest();
      allComps.forEach(function(c,i){
        var p=xy(c.pfx_x,c.pfx_z);
        var r=closest[i]?7:5;
        var dx=mx-p.x, dy=my-p.y;
        if(dx*dx+dy*dy<=r*r){hit=c; hitType='comp';}
      });
    }

    if(hit){
      if(hitType==='arsenal'){
        var td=tunnelData[hit.pt];
        var lines=['<div style="font-weight:700;color:'+hit.color+';">'+hit.name+'</div>'];
        if(td){
          lines.push('<div style="margin-top:4px;font-size:10px;line-height:1.5;">');
          lines.push('<div>Late break ratio: <b>'+td.curr_lbr.toFixed(2)+'</b> &rarr; <b>'+td.opt_lbr.toFixed(2)+'</b></div>');
          lines.push('<div>Tunnel pt sep: <b>'+td.curr_tun.toFixed(1)+'"</b> &rarr; <b>'+td.opt_tun.toFixed(1)+'"</b></div>');
          lines.push('<div>Plate sep: <b>'+td.curr_plate.toFixed(1)+'"</b> &rarr; <b>'+td.opt_plate.toFixed(1)+'"</b></div>');
          lines.push('</div>');
          if(hit.is_pair){
            lines.push('<div style="margin-top:4px;color:#1D9E75;font-size:9px;">\u2605 tunnel pair</div>');
          }
        } else {
          lines.push('<div style="margin-top:2px;font-size:10px;color:#aaa;">not a tunnel pair</div>');
        }
        tt.innerHTML=lines.join('');
      } else {
        tt.textContent=hit.name+(hit.is_mirrored?' (mirrored)':'');
      }
      tt.style.display='block';
      tt.style.left=(e.clientX-rect.left+12)+'px';
      tt.style.top=(e.clientY-rect.top-30)+'px';
    } else {
      tt.style.display='none';
    }
  });
  canv.addEventListener('mouseleave',function(){tt.style.display='none';});

  setup(); draw();
  var tPT;
  window.addEventListener('resize',function(){clearTimeout(tPT);tPT=setTimeout(function(){setup();draw();},80);});
})();
</script>
"""
    js = js.replace('COMP_JSON', comp_json)
    js = js.replace('ARSENAL_JSON', arsenal_json)
    js = js.replace('TUNNEL_JSON', tunnel_json)
    js = js.replace('PT', pt).replace('CX_F', str(cx_f)).replace('CZ_F', str(cz_f))
    js = js.replace('OX_F', str(ox_f)).replace('OZ_F', str(oz_f))
    js = js.replace('tPT', f't_{pt}')

    components.html(html + js, height=800, scrolling=False)


def render_tunneling_section(selected_pt, selected_pt_info, profile,
                              opt_pfx_x, opt_pfx_z, opt_velo):
    """
    Render the tunneling analysis section.
    Shows current and optimized tunneling for each pair the selected pitch makes
    with the other pitches in the pitcher's arsenal.
    """
    from tunneling import compute_tunnel_metrics, make_pitch_dict_from_optimized

    # Build pitch dict for selected pitch — both current and optimized
    sel_fixed = selected_pt_info['fixed']
    sel_opt   = selected_pt_info['optimizable']
    sel_semi  = selected_pt_info['semi_fixed']

    # Need all trajectory fields
    required = ['release_pos_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    if not all(sel_fixed.get(k) is not None for k in required):
        st.info("Tunneling analysis requires updated profile data with trajectory features. "
                "Rerun the notebook to regenerate `pitcher_stuff_profiles.pkl`.")
        return

    sel_current = {
        'release_pos_x': sel_fixed['release_pos_x'],
        'release_pos_y': sel_fixed['release_pos_y'],
        'release_pos_z': sel_fixed['release_pos_z'],
        'vx0': sel_fixed['vx0'], 'vy0': sel_fixed['vy0'], 'vz0': sel_fixed['vz0'],
        'ax':  sel_fixed['ax'],  'ay':  sel_fixed['ay'],  'az':  sel_fixed['az'],
    }
    sel_optimized = make_pitch_dict_from_optimized(
        opt_pfx_x, opt_pfx_z, opt_velo,
        sel_fixed['release_pos_x'], sel_fixed['release_pos_y'], sel_fixed['release_pos_z'],
        vx0_actual=sel_fixed['vx0'], vy0_actual=sel_fixed['vy0'],
        vz0_actual=sel_fixed['vz0'], ay_actual=sel_fixed['ay'],
    )

    # Build pairings with each other pitch (filtered by physical tunnel pair criteria)
    pairings = []
    for other_pt, other_info in profile['pitches'].items():
        if other_pt == selected_pt:
            continue
        ofx = other_info['fixed']
        if not all(ofx.get(k) is not None for k in required):
            continue
        # Filter by tunnel pair physical criteria (velo gap, release point similarity)
        if not is_tunnel_pair(selected_pt_info, other_info):
            continue
        other_pitch = {
            'release_pos_x': ofx['release_pos_x'],
            'release_pos_y': ofx['release_pos_y'],
            'release_pos_z': ofx['release_pos_z'],
            'vx0': ofx['vx0'], 'vy0': ofx['vy0'], 'vz0': ofx['vz0'],
            'ax':  ofx['ax'],  'ay':  ofx['ay'],  'az':  ofx['az'],
        }
        curr_metrics = compute_tunnel_metrics(sel_current, other_pitch)
        opt_metrics  = compute_tunnel_metrics(sel_optimized, other_pitch)
        if curr_metrics is None or opt_metrics is None:
            continue
        pairings.append({
            'other_pt':     other_pt,
            'other_name':   pitch_name(other_pt),
            'other_color':  pitch_color(other_pt),
            'curr':         curr_metrics,
            'opt':          opt_metrics,
        })

    if not pairings:
        st.info("No tunnel pairs for this pitch under research criteria (velocity differential 2-12 mph "
                "and release point within 2 inches). The flight path chart below still shows all pitches.")
        return

    # Sort by best current tunneling score
    pairings.sort(key=lambda p: p['curr']['late_break_ratio'], reverse=True)

    # Header
    st.markdown(
        '<div style="margin:1.2rem 0 0.6rem;display:flex;align-items:center;gap:10px;">'
        f'<span style="font-family:Barlow Condensed,sans-serif;font-size:1rem;font-weight:800;'
        f'color:#3a352c;text-transform:uppercase;letter-spacing:0.04em;">Tunneling Analysis</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:#8c857a;">'
        f'— how {pitch_name(selected_pt).lower()} pairs with other pitches</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # Card row
    n = len(pairings)
    cols = st.columns(n)
    for col, pair in zip(cols, pairings):
        curr_lbr = pair['curr']['late_break_ratio']
        opt_lbr  = pair['opt']['late_break_ratio']
        delta    = round(opt_lbr - curr_lbr, 2)
        curr_tun = pair['curr']['tunnel_sep_in']
        opt_tun  = pair['opt']['tunnel_sep_in']
        curr_plate = pair['curr']['plate_sep_in']
        opt_plate  = pair['opt']['plate_sep_in']

        # Color the delta: green = better, red = worse, grey = ~same
        if delta > 0.05:
            d_col = '#1D9E75'; d_sign = '+'
            d_label = 'better tunneling'
        elif delta < -0.05:
            d_col = '#ef4444'; d_sign = ''
            d_label = 'worse tunneling'
        else:
            d_col = '#6b7280'; d_sign = ('+' if delta >= 0 else '')
            d_label = 'similar'

        # Late-break-ratio quality color
        def lbr_color(r):
            if r >= 4:   return '#1D9E75'  # excellent
            if r >= 2.5: return '#84cc16'  # good
            if r >= 1.5: return '#f59e0b'  # average
            return '#ef4444'                # poor

        col.markdown(
            f'<div style="background:#faf8f4;border:1px solid rgba(255,255,255,0.06);'
            f'border-top:3px solid {pair["other_color"]};border-radius:0 0 8px 8px;'
            f'padding:0.85rem 1rem;height:100%;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:0.6rem;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{pair["other_color"]};"></span>'
            f'<span style="font-family:Barlow Condensed,sans-serif;font-size:0.78rem;font-weight:700;'
            f'color:{pair["other_color"]};text-transform:uppercase;letter-spacing:0.04em;">'
            f'vs {pair["other_name"]}</span>'
            f'</div>'
            # Late-break ratio big number
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
            f'margin-bottom:0.5rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<span style="font-family:Barlow Condensed,sans-serif;font-size:1.4rem;font-weight:800;'
            f'color:{lbr_color(curr_lbr)};line-height:1;">{curr_lbr:.1f}</span>'
            f'<span style="font-size:0.7rem;color:#8c857a;">→</span>'
            f'<span style="font-family:Barlow Condensed,sans-serif;font-size:1.4rem;font-weight:800;'
            f'color:{lbr_color(opt_lbr)};line-height:1;">{opt_lbr:.1f}</span>'
            f'</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.55rem;color:#8c857a;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">late break ratio</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;font-weight:600;'
            f'color:{d_col};">{d_sign}{delta}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.55rem;color:{d_col};'
            f'opacity:0.8;margin-top:2px;">{d_label}</div>'
            f'</div>'
            f'</div>'
            # Detail rows
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
            f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">tunnel pt sep</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#3a352c;">'
            f'{curr_tun:.1f}" → {opt_tun:.1f}"</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
            f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#8c857a;">plate sep</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#3a352c;">'
            f'{curr_plate:.1f}" → {opt_plate:.1f}"</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def is_tunnel_pair(pitch_a_info, pitch_b_info):
    """
    Determine whether two pitches can tunnel together based on physical criteria
    from tunneling research (Long/Pavlidis/Judge 2017, Driveline pitch design).

    Criteria:
      1. Velocity differential under 12 mph (keeps commit timing coupled)
      2. Release point within 2 inches in both x and z
      3. Minimum velocity difference of 2 mph (otherwise same pitch type)

    Returns True if the pair satisfies all criteria.
    """
    a_fix = pitch_a_info.get('fixed', {})
    b_fix = pitch_b_info.get('fixed', {})
    a_sem = pitch_a_info.get('semi_fixed', {})
    b_sem = pitch_b_info.get('semi_fixed', {})

    velo_a = a_sem.get('velo_mean')
    velo_b = b_sem.get('velo_mean')
    if velo_a is None or velo_b is None:
        return False

    velo_diff = abs(velo_a - velo_b)
    if velo_diff > 12 or velo_diff < 2:
        return False

    rpx_a = a_fix.get('release_pos_x')
    rpx_b = b_fix.get('release_pos_x')
    rpz_a = a_fix.get('release_pos_z')
    rpz_b = b_fix.get('release_pos_z')
    if None in (rpx_a, rpx_b, rpz_a, rpz_b):
        return False

    # 2 inches = 0.167 ft
    if abs(rpx_a - rpx_b) > 0.167 or abs(rpz_a - rpz_b) > 0.167:
        return False

    return True


def compute_arsenal_improvements(profile, norm_tables, stuff_models):
    """
    For each pitch in the arsenal, optimize and check whether the optimal shape
    improves BOTH Stuff+ AND tunneling with at least one other pitch.
    Returns dict {pt: {'improves_both': bool, 'opt_shape': (x,z), 'opt_grades': {...}}}.
    """
    from tunneling import compute_tunnel_metrics, make_pitch_dict_from_optimized

    traj_keys = ['release_pos_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    pitches = profile.get('pitches', {})
    result = {}

    # Pre-build pitch dicts for all pitches that have trajectory data
    pitch_dicts = {}
    for pt, info in pitches.items():
        fixed = info.get('fixed', {})
        if all(fixed.get(k) is not None for k in traj_keys):
            pitch_dicts[pt] = {
                'release_pos_x': fixed['release_pos_x'],
                'release_pos_y': fixed['release_pos_y'],
                'release_pos_z': fixed['release_pos_z'],
                'vx0': fixed['vx0'], 'vy0': fixed['vy0'], 'vz0': fixed['vz0'],
                'ax':  fixed['ax'],  'ay':  fixed['ay'],  'az':  fixed['az'],
            }

    for pt, info in pitches.items():
        if pt not in pitch_dicts:
            result[pt] = {'improves_both': False}
            continue

        try:
            opt_pfx_x, opt_pfx_z, _, opt_grades = run_optimizer(
                pt, info, norm_tables, stuff_models, stand='C'
            )
        except Exception:
            result[pt] = {'improves_both': False}
            continue

        # Require meaningful Stuff+ improvement
        curr_sp = info.get('grades', {}).get('stuff_plus', 100)
        opt_sp  = opt_grades.get('stuff_plus', 100)
        if opt_sp <= curr_sp + 0.5:
            result[pt] = {'improves_both': False, 'opt_sp': opt_sp, 'curr_sp': curr_sp}
            continue

        # Build optimized pitch dict
        velo = info.get('semi_fixed', {}).get('velo_mean', 90)
        fixed = info['fixed']
        optimized = make_pitch_dict_from_optimized(
            opt_pfx_x, opt_pfx_z, velo,
            fixed['release_pos_x'], fixed['release_pos_y'], fixed['release_pos_z'],
            vx0_actual=fixed['vx0'], vy0_actual=fixed['vy0'],
            vz0_actual=fixed['vz0'], ay_actual=fixed['ay'],
        )
        current = pitch_dicts[pt]

        # Check tunneling against each PAIRED pitch (filtered by physical criteria)
        improves_any = False
        improved_pairs = []
        all_pair_results = []  # for tooltip display later
        for other_pt, other in pitch_dicts.items():
            if other_pt == pt: continue
            # Only consider physically valid tunnel pairs
            if not is_tunnel_pair(info, pitches[other_pt]):
                continue
            cm = compute_tunnel_metrics(current, other)
            om = compute_tunnel_metrics(optimized, other)
            if cm and om:
                all_pair_results.append({
                    'pair': other_pt,
                    'curr_lbr': cm['late_break_ratio'],
                    'opt_lbr':  om['late_break_ratio'],
                })
                if om['late_break_ratio'] > cm['late_break_ratio'] + 0.1:
                    improves_any = True
                    improved_pairs.append(other_pt)

        result[pt] = {
            'improves_both':  improves_any,
            'improved_pairs': improved_pairs,
            'all_pairs':      all_pair_results,
            'opt_sp':         opt_sp,
            'curr_sp':        curr_sp,
        }

    return result


def render_trajectory_chart(selected_pt, sel_pt_info, profile,
                             opt_pfx_x, opt_pfx_z, opt_velo):
    """
    Render an isometric 3D trajectory visualization showing the flight paths of:
    - Selected pitch (current shape)
    - Selected pitch (optimized shape)
    - Each tunnel-pair pitch in the arsenal

    Camera positioned behind and slightly above pitcher, looking down the line
    toward the plate.
    """
    import streamlit.components.v1 as components
    import json, numpy as np
    from tunneling import trajectory_at_y, make_pitch_dict_from_optimized

    traj_keys = ['release_pos_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    sel_fix = sel_pt_info.get('fixed', {})
    if not all(sel_fix.get(k) is not None for k in traj_keys):
        return

    # Build trajectories — sample along the flight path
    def sample_trajectory(pitch_dict, n=40):
        rpy = pitch_dict['release_pos_y']
        # Sample y values from release to plate
        ys = np.linspace(rpy, 1.417, n)
        pts = []
        for y in ys:
            x, z = trajectory_at_y(
                pitch_dict['release_pos_x'], pitch_dict['release_pos_y'], pitch_dict['release_pos_z'],
                pitch_dict['vx0'], pitch_dict['vy0'], pitch_dict['vz0'],
                pitch_dict['ax'], pitch_dict['ay'], pitch_dict['az'],
                float(y)
            )
            if x is not None:
                pts.append({'x': float(x), 'y': float(y), 'z': float(z)})
        return pts

    sel_current_dict = {
        'release_pos_x': sel_fix['release_pos_x'],
        'release_pos_y': sel_fix['release_pos_y'],
        'release_pos_z': sel_fix['release_pos_z'],
        'vx0': sel_fix['vx0'], 'vy0': sel_fix['vy0'], 'vz0': sel_fix['vz0'],
        'ax':  sel_fix['ax'],  'ay':  sel_fix['ay'],  'az':  sel_fix['az'],
    }
    sel_optimized_dict = make_pitch_dict_from_optimized(
        opt_pfx_x, opt_pfx_z, opt_velo,
        sel_fix['release_pos_x'], sel_fix['release_pos_y'], sel_fix['release_pos_z'],
        vx0_actual=sel_fix['vx0'], vy0_actual=sel_fix['vy0'],
        vz0_actual=sel_fix['vz0'], ay_actual=sel_fix['ay'],
    )

    # Identify tunnel pairs for the selected pitch
    trajectories = [
        {
            'label': pitch_name(selected_pt) + ' (current)',
            'color': '#3b82f6',
            'style': 'solid',
            'points': sample_trajectory(sel_current_dict),
            'pt': selected_pt,
            'kind': 'current',
        },
        {
            'label': pitch_name(selected_pt) + ' (optimized)',
            'color': '#f59e0b',
            'style': 'dashed',
            'points': sample_trajectory(sel_optimized_dict),
            'pt': selected_pt,
            'kind': 'optimized',
        },
    ]

    pitches_added = 0
    pair_count = 0
    for other_pt, other_info in profile['pitches'].items():
        if other_pt == selected_pt:
            continue
        ofx = other_info.get('fixed', {})
        if not all(ofx.get(k) is not None for k in traj_keys):
            continue
        is_pair = is_tunnel_pair(sel_pt_info, other_info)
        other_dict = {
            'release_pos_x': ofx['release_pos_x'],
            'release_pos_y': ofx['release_pos_y'],
            'release_pos_z': ofx['release_pos_z'],
            'vx0': ofx['vx0'], 'vy0': ofx['vy0'], 'vz0': ofx['vz0'],
            'ax':  ofx['ax'],  'ay':  ofx['ay'],  'az':  ofx['az'],
        }
        trajectories.append({
            'label': pitch_name(other_pt) + ('' if is_pair else ' (not a tunnel pair)'),
            'color': pitch_color(other_pt),
            'style': 'solid' if is_pair else 'dotted',
            'points': sample_trajectory(other_dict),
            'pt': other_pt,
            'kind': 'pair' if is_pair else 'nonpair',
        })
        pitches_added += 1
        if is_pair:
            pair_count += 1

    if pitches_added == 0:
        # No other pitches with trajectory data — show just the selected pitch's
        # current vs optimized paths (still useful on its own)
        pass

    # Tunnel point (y from plate) — average release_y - 23
    avg_rpy = sel_fix['release_pos_y']
    tunnel_y = avg_rpy - 23.0

    # Dynamic world Y_MAX — the farthest release point among shown pitches plus margin,
    # so no trajectory starts outside the rendered volume (fixes pitches drawn past zone)
    max_release_y = avg_rpy
    for t in trajectories:
        if t['points']:
            max_release_y = max(max_release_y, t['points'][0]['y'])
    world_y_max = float(np.ceil(max_release_y + 2))

    # Dynamic X and Z bounds — fit to the actual trajectory extents so pitches with
    # big horizontal sweep (sliders/curves out to -4 ft) or unusual release heights
    # don't spill outside the rendered box. Previously hardcoded ±3 / 0-7, which
    # clipped wide breaking balls (showed up with custom-CSV pitchers).
    all_x, all_z = [], []
    for t in trajectories:
        for pnt in t['points']:
            all_x.append(pnt['x'])
            all_z.append(pnt['z'])
    if all_x:
        x_lo = float(np.floor(min(all_x) - 0.5))
        x_hi = float(np.ceil(max(all_x) + 0.5))
        # keep symmetric-ish around 0 so the plate stays visually centered
        x_abs = max(abs(x_lo), abs(x_hi), 3.0)
        world_x_min, world_x_max = -x_abs, x_abs
        world_z_min = float(min(0.0, np.floor(min(all_z))))
        world_z_max = float(max(7.0, np.ceil(max(all_z) + 0.5)))
    else:
        world_x_min, world_x_max, world_z_min, world_z_max = -3.0, 3.0, 0.0, 7.0

    traj_json = json.dumps(trajectories)

    html = f"""
<div style="background:#faf8f4;border:1px solid #e0dbd0;border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:0.75rem;font-family:sans-serif;width:100%;box-sizing:border-box;max-width:100%;overflow-x:hidden;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <div>
      <span style="font-size:14px;font-weight:700;color:#3a352c;">Flight Path Visualization</span>
      <span style="font-size:12px;color:#888;margin-left:10px;">3D trajectories with decision point</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,240px);gap:14px;align-items:start;max-width:100%;box-sizing:border-box;">
    <div style="position:relative;width:100%;height:500px;min-width:0;">
      <canvas id="tc_PT_TRAJ" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
      <div id="tt_PT_TRAJ" style="position:absolute;display:none;background:rgba(0,0,0,0.82);color:white;font-size:11px;padding:5px 9px;border-radius:5px;pointer-events:none;white-space:nowrap;z-index:10;"></div>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;min-width:0;">
      <div style="font-size:11px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.06em;">
        Separation at Decision Point
      </div>
      <div style="font-size:10px;color:#888;margin-bottom:4px;">
        Other pitches' positions relative to current selected pitch (inches)
      </div>
      <div style="position:relative;width:100%;aspect-ratio:1;background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.4);border-radius:6px;overflow:hidden;">
        <canvas id="cs_PT_TRAJ" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
      </div>
      <div id="cs_legend_PT_TRAJ" style="font-size:10px;color:#666;line-height:1.5;"></div>
    </div>
  </div>
  <div id="legend_PT_TRAJ" style="display:flex;gap:18px;margin-top:8px;flex-wrap:wrap;align-items:center;font-size:12px;color:#555;"></div>
</div>
"""
    html = html.replace('PT_TRAJ', selected_pt + '_traj')

    js = """
<script>
(function(){
  var canv=document.getElementById('tc_TRAJID');
  var ctx=canv.getContext('2d');
  var tt=document.getElementById('tt_TRAJID');
  var legend=document.getElementById('legend_TRAJID');
  var W,H,DPR;
  var trajectories=TRAJ_JSON;
  var tunnelY=TUNNEL_Y;

  // Camera setup — behind home plate, slightly elevated and offset to one side.
  // This is the oblique "broadcast-style" angle. Rotation-based projection:
  // yaw turns the scene around vertical, pitch tilts it down, and a mild
  // perspective scale shrinks far objects.
  var YAW   = -0.35;  // slight 1B-side offset
  var PITCH = 0.20;   // downward tilt (~11°)

  // World extents
  var Y_MIN = 0;
  var Y_MAX = WORLD_Y_MAX;
  var Z_MIN = WORLD_Z_MIN;
  var Z_MAX = WORLD_Z_MAX;
  var X_MIN = WORLD_X_MIN;
  var X_MAX = WORLD_X_MAX;

  function project(wx, wy, wz){
    var cy=Math.cos(YAW), sy=Math.sin(YAW);
    var x1 = wx*cy - wy*sy;
    var y1 = wx*sy + wy*cy;
    var z1 = wz;
    var cp=Math.cos(PITCH), sp=Math.sin(PITCH);
    var y2 = y1*cp - z1*sp;
    var z2 = y1*sp + z1*cp;
    return {sx: x1, sy: -z2, depth: y2};
  }

  function perspScale(depth){
    var camDist = 30;
    return camDist / (camDist + depth);
  }

  function worldToScreen(wx, wy, wz){
    var p = project(wx, wy, wz);
    var ps = perspScale(p.depth);
    return {
      x: CX + p.sx * SC * ps,
      y: CY + p.sy * SC * ps,
      d: p.depth
    };
  }

  var CX, CY, SC;

  function setup(){
    DPR=window.devicePixelRatio||1;
    W=canv.offsetWidth; H=canv.offsetHeight;
    canv.width=W*DPR; canv.height=H*DPR;
    ctx.setTransform(DPR,0,0,DPR,0,0);

    // Fit the view to the full world bounding box (all 8 corners, with perspective
    // applied). This frames the whole scene — ground grid, strike zone, decision
    // plane, and the full flight from release to plate — generously in the canvas,
    // which is the framing the visualization was designed around. (Fitting to just
    // the trajectories zoomed in too far and cropped out the surrounding scene.)
    var corners = [
      [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MIN, Z_MIN], [X_MIN, Y_MAX, Z_MIN], [X_MAX, Y_MAX, Z_MIN],
      [X_MIN, Y_MIN, Z_MAX], [X_MAX, Y_MIN, Z_MAX], [X_MIN, Y_MAX, Z_MAX], [X_MAX, Y_MAX, Z_MAX],
    ];
    var minSX=Infinity, maxSX=-Infinity, minSY=Infinity, maxSY=-Infinity;
    corners.forEach(function(c){
      var p = project(c[0], c[1], c[2]);
      var ps = perspScale(p.depth);
      var psx = p.sx * ps, psy = p.sy * ps;
      if(psx<minSX) minSX=psx;
      if(psx>maxSX) maxSX=psx;
      if(psy<minSY) minSY=psy;
      if(psy>maxSY) maxSY=psy;
    });

    var pad = 30;
    var rangeX = (maxSX - minSX) || 1;
    var rangeY = (maxSY - minSY) || 1;
    SC = Math.min((W - 2*pad) / rangeX, (H - 2*pad) / rangeY) * 0.95;
    CX = W/2 - (minSX + maxSX)/2 * SC;
    CY = H/2 - (minSY + maxSY)/2 * SC;
  }

  function drawGround(){
    // Draw a ground plane grid
    ctx.strokeStyle='rgba(180,200,220,0.3)';
    ctx.lineWidth=0.5;
    // Lines along x (parallel to plate)
    for(var y=Y_MIN; y<=Y_MAX; y+=10){
      var a = worldToScreen(X_MIN, y, 0);
      var b = worldToScreen(X_MAX, y, 0);
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    }
    // Lines along y (depth)
    for(var x=Math.ceil(X_MIN); x<=Math.floor(X_MAX); x+=1){
      var a = worldToScreen(x, Y_MIN, 0);
      var b = worldToScreen(x, Y_MAX, 0);
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    }
  }

  function drawStrikeZone(){
    // Strike zone: roughly 1.5 to 3.5 ft vertical, -0.83 to 0.83 ft horizontal
    var corners=[
      worldToScreen(-0.83, 1.417, 1.5),
      worldToScreen( 0.83, 1.417, 1.5),
      worldToScreen( 0.83, 1.417, 3.5),
      worldToScreen(-0.83, 1.417, 3.5),
    ];
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for(var i=1; i<4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.fillStyle='rgba(100,180,220,0.06)';
    ctx.fill();
    ctx.strokeStyle='rgba(80,120,180,0.45)';
    ctx.lineWidth=1.2;
    ctx.stroke();

    // Home plate (pentagon, on the ground)
    var plate=[
      worldToScreen(-0.71, 1.417, 0),
      worldToScreen( 0.71, 1.417, 0),
      worldToScreen( 0.71, 0.708, 0),
      worldToScreen( 0,    0,     0),
      worldToScreen(-0.71, 0.708, 0),
    ];
    ctx.beginPath();
    ctx.moveTo(plate[0].x, plate[0].y);
    for(var i=1; i<5; i++) ctx.lineTo(plate[i].x, plate[i].y);
    ctx.closePath();
    ctx.fillStyle='rgba(255,255,255,0.92)';
    ctx.strokeStyle='rgba(120,120,140,0.8)';
    ctx.lineWidth=1; ctx.fill(); ctx.stroke();
  }

  function drawDecisionPlane(){
    // Draw vertical translucent plane at decision point y
    var plane=[
      worldToScreen(-3, tunnelY, 0),
      worldToScreen( 3, tunnelY, 0),
      worldToScreen( 3, tunnelY, 7),
      worldToScreen(-3, tunnelY, 7),
    ];
    ctx.beginPath();
    ctx.moveTo(plane[0].x, plane[0].y);
    for(var i=1; i<4; i++) ctx.lineTo(plane[i].x, plane[i].y);
    ctx.closePath();
    ctx.fillStyle='rgba(245,158,11,0.10)';
    ctx.strokeStyle='rgba(245,158,11,0.55)';
    ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.fill(); ctx.stroke(); ctx.setLineDash([]);

    // Label
    var labelP = worldToScreen(3.2, tunnelY, 5.5);
    ctx.fillStyle='#b45309';
    ctx.font='bold 10px sans-serif';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText('DECISION POINT', labelP.x, labelP.y);
    ctx.font='9px sans-serif';
    ctx.fillStyle='#92400e';
    ctx.fillText('~23 ft from release', labelP.x, labelP.y+12);
  }

  function findTunnelPoint(pts){
    // Find the point in the trajectory closest to the tunnel_y plane
    var best=null, bestDist=Infinity;
    for(var i=0; i<pts.length; i++){
      var d = Math.abs(pts[i].y - tunnelY);
      if(d < bestDist){bestDist=d; best=pts[i];}
    }
    return best;
  }

  function drawTrajectory(traj){
    if(traj.points.length<2) return;
    var isNonpair = (traj.kind === 'nonpair');
    ctx.save();
    if(isNonpair) ctx.globalAlpha = 0.4;  // dim non-pair pitches
    ctx.strokeStyle=traj.color;
    ctx.lineWidth = isNonpair ? 1.5 : 2.5;
    if(traj.style==='dashed') ctx.setLineDash([6,4]);
    else if(traj.style==='dotted') ctx.setLineDash([2,4]);

    ctx.beginPath();
    var first = worldToScreen(traj.points[0].x, traj.points[0].y, traj.points[0].z);
    ctx.moveTo(first.x, first.y);
    for(var i=1; i<traj.points.length; i++){
      var p = worldToScreen(traj.points[i].x, traj.points[i].y, traj.points[i].z);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Release point dot
    ctx.beginPath();
    ctx.arc(first.x, first.y, 4, 0, Math.PI*2);
    ctx.fillStyle=traj.color; ctx.fill();
    ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();

    // Plate end dot
    var last = traj.points[traj.points.length-1];
    var lastP = worldToScreen(last.x, last.y, last.z);
    ctx.beginPath();
    ctx.arc(lastP.x, lastP.y, 5, 0, Math.PI*2);
    ctx.fillStyle=traj.color; ctx.fill();
    ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();

    // Tunnel-point marker (small ring) — only for pairs and the selected pitch
    if(!isNonpair){
      var tp = findTunnelPoint(traj.points);
      if(tp){
        var tpP = worldToScreen(tp.x, tp.y, tp.z);
        ctx.beginPath();
        ctx.arc(tpP.x, tpP.y, 6, 0, Math.PI*2);
        ctx.strokeStyle=traj.color; ctx.lineWidth=2; ctx.stroke();
        ctx.beginPath();
        ctx.arc(tpP.x, tpP.y, 2, 0, Math.PI*2);
        ctx.fillStyle=traj.color; ctx.fill();
      }
    }
    ctx.restore();
  }

  function buildLegend(){
    var html='';
    trajectories.forEach(function(t){
      var border;
      if(t.style==='dashed') border = 'border-bottom:2px dashed '+t.color;
      else if(t.style==='dotted') border = 'border-bottom:2px dotted '+t.color;
      else border = 'border-bottom:2px solid '+t.color;
      var op = (t.kind==='nonpair') ? 'opacity:0.5;' : '';
      html += '<div style="display:flex;align-items:center;gap:6px;'+op+'">';
      html += '<div style="width:18px;'+border+';"></div>';
      html += '<span>'+t.label+'</span>';
      html += '</div>';
    });
    legend.innerHTML = html;
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    drawGround();
    drawStrikeZone();
    drawDecisionPlane();
    // Sort trajectories so paired pitches draw on top (visually emphasized)
    var sorted = trajectories.slice();
    sorted.forEach(drawTrajectory);
  }

  // ── Cross-section panel — head-on view at decision point ─────────────────
  var csCanv = document.getElementById('cs_TRAJID');
  var csCtx = csCanv.getContext('2d');
  var csLegend = document.getElementById('cs_legend_TRAJID');
  var csW, csH, csCX, csCY, csSC;

  function csSetup(){
    var dpr = window.devicePixelRatio || 1;
    csW = csCanv.offsetWidth;
    csH = csCanv.offsetHeight;
    csCanv.width = csW * dpr;
    csCanv.height = csH * dpr;
    csCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    csCX = csW / 2;
    csCY = csH / 2;
  }

  function findTunnelPoint2(pts){
    var best = null, bestDist = Infinity;
    for(var i = 0; i < pts.length; i++){
      var d = Math.abs(pts[i].y - tunnelY);
      if(d < bestDist){ bestDist = d; best = pts[i]; }
    }
    return best;
  }

  function csDraw(){
    csCtx.clearRect(0, 0, csW, csH);

    // Find the current selected pitch — it's the anchor at (0, 0)
    var refTraj = null, optTraj = null;
    trajectories.forEach(function(t){
      if(t.kind === 'current') refTraj = t;
      if(t.kind === 'optimized') optTraj = t;
    });
    if(!refTraj) return;
    var refTp = findTunnelPoint2(refTraj.points);
    if(!refTp) return;

    // Compute all pitch offsets from the anchor (in inches)
    // Only include the selected pitch (current + optimized) and valid tunnel pairs —
    // the decision-point separation analysis is meaningful only for pairs.
    var offsets = [];
    trajectories.forEach(function(t){
      if(t.kind === 'nonpair') return;  // skip non-pairs in cross-section
      var tp = findTunnelPoint2(t.points);
      if(!tp) return;
      offsets.push({
        traj: t,
        dx: (tp.x - refTp.x) * 12,  // inches, x flipped for pitcher POV below
        dz: (tp.z - refTp.z) * 12,
      });
    });

    // Auto-zoom: find max absolute offset, add 30% padding, min 3"
    var maxOff = 3;
    offsets.forEach(function(o){
      var m = Math.max(Math.abs(o.dx), Math.abs(o.dz));
      if(m > maxOff) maxOff = m;
    });
    maxOff = Math.ceil(maxOff * 1.3);
    if(maxOff < 3) maxOff = 3;

    // Scale: maxOff inches → half of inner area. Padding must reserve room for
    // the marker (radius 6 + glow ring ~5) AND its adjacent text label (~22px),
    // or a pitch at the edge of the data range draws its dot/label past the canvas
    // edge. 34px clears all of that on every side.
    var pad = 34;
    var innerR = Math.min(csW, csH) / 2 - pad;
    if(innerR < 20) innerR = 20;
    var csSC2 = innerR / maxOff;  // pixels per inch

    function csXY(dx_in, dz_in){
      // Pitcher POV: positive pfx_x (1B side) on the LEFT (negate)
      return { x: csCX - dx_in * csSC2, y: csCY - dz_in * csSC2 };
    }

    // Determine gridline spacing based on zoom level
    var gridStep = 1;
    if(maxOff > 8)  gridStep = 2;
    if(maxOff > 16) gridStep = 4;
    if(maxOff > 30) gridStep = 8;

    // Draw gridlines
    csCtx.strokeStyle = 'rgba(150,160,180,0.18)';
    csCtx.lineWidth = 0.5;
    for(var g = gridStep; g <= maxOff; g += gridStep){
      // Concentric squares (rectangles centered on origin)
      var p1 = csXY(g, g);
      var p2 = csXY(-g, -g);
      csCtx.beginPath();
      csCtx.rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      csCtx.stroke();
    }

    // Axis crosshairs
    csCtx.strokeStyle = 'rgba(100,116,139,0.35)';
    csCtx.lineWidth = 0.8;
    csCtx.beginPath();
    csCtx.moveTo(csCX, 4); csCtx.lineTo(csCX, csH - 4);
    csCtx.moveTo(4, csCY); csCtx.lineTo(csW - 4, csCY);
    csCtx.stroke();

    // Tick marks and labels on the axes
    csCtx.fillStyle = '#666';
    csCtx.font = '8px sans-serif';
    csCtx.textAlign = 'center'; csCtx.textBaseline = 'top';
    for(var g = gridStep; g <= maxOff; g += gridStep){
      var pR = csXY(-g, 0);  // right side (1B side flipped)
      var pL = csXY(g, 0);   // left side
      csCtx.fillText(g + '"', pR.x, csCY + 4);
      csCtx.fillText(g + '"', pL.x, csCY + 4);
    }
    csCtx.textAlign = 'right'; csCtx.textBaseline = 'middle';
    for(var g = gridStep; g <= maxOff; g += gridStep){
      var pU = csXY(0, g);   // up
      var pD = csXY(0, -g);  // down
      csCtx.fillText(g + '"', csCX - 4, pU.y);
      csCtx.fillText(g + '"', csCX - 4, pD.y);
    }

    // Direction labels at edges
    csCtx.font = '9px sans-serif';
    csCtx.fillStyle = '#94a3b8';
    csCtx.textAlign = 'left'; csCtx.textBaseline = 'middle';
    csCtx.fillText('3B', 6, csCY - 7);
    csCtx.textAlign = 'right';
    csCtx.fillText('1B', csW - 6, csCY - 7);
    csCtx.textAlign = 'center'; csCtx.textBaseline = 'top';
    csCtx.fillText('UP', csCX + 10, 2);
    csCtx.textBaseline = 'bottom';
    csCtx.fillText('DOWN', csCX + 18, csH - 2);

    // Draw each pitch as an offset marker
    offsets.forEach(function(o){
      var p = csXY(o.dx, o.dz);
      var isOpt = o.traj.kind === 'optimized';
      var isCurrent = o.traj.kind === 'current';
      var r = 6;

      // Subtle glow for paired pitches
      if(o.traj.kind === 'pair'){
        csCtx.beginPath();
        csCtx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
        csCtx.fillStyle = o.traj.color + '22';
        csCtx.fill();
      }

      if(isOpt){
        // Triangle for optimized
        csCtx.beginPath();
        csCtx.moveTo(p.x, p.y - r);
        csCtx.lineTo(p.x + r * 0.87, p.y + r * 0.5);
        csCtx.lineTo(p.x - r * 0.87, p.y + r * 0.5);
        csCtx.closePath();
        csCtx.fillStyle = o.traj.color;
        csCtx.fill();
        csCtx.strokeStyle = 'white';
        csCtx.lineWidth = 1.5;
        csCtx.stroke();
      } else {
        // Circle (current selected + pairs)
        csCtx.beginPath();
        csCtx.arc(p.x, p.y, r, 0, Math.PI * 2);
        csCtx.fillStyle = o.traj.color;
        csCtx.fill();
        csCtx.strokeStyle = 'white';
        csCtx.lineWidth = isCurrent ? 2 : 1.5;
        csCtx.stroke();
      }

      // Pitch type label next to dot
      var labelX = p.x + r + 4;
      var labelY = p.y;
      // If too close to right edge, put the label on the left side of the dot
      if(labelX > csW - 26) {
        labelX = p.x - r - 4;
        csCtx.textAlign = 'right';
      } else {
        csCtx.textAlign = 'left';
      }
      csCtx.font = 'bold 9px sans-serif';
      csCtx.fillStyle = o.traj.color;
      csCtx.textBaseline = 'middle';
      csCtx.fillText(o.traj.pt, labelX, labelY);
    });

    // Center "current" label
    csCtx.font = '8px sans-serif';
    csCtx.fillStyle = '#94a3b8';
    csCtx.textAlign = 'center';
    csCtx.textBaseline = 'top';
    csCtx.fillText('= current ' + refTraj.pt, csCX, csCY + 9);

    // Build text legend showing separation distances
    var lines = [];
    if(refTraj && optTraj){
      var optTp = findTunnelPoint2(optTraj.points);
      trajectories.forEach(function(t){
        if(t.kind !== 'pair') return;
        var tp = findTunnelPoint2(t.points);
        if(!tp || !refTp || !optTp) return;
        var dxC = (tp.x - refTp.x) * 12;
        var dzC = (tp.z - refTp.z) * 12;
        var sepC = Math.sqrt(dxC*dxC + dzC*dzC);
        var dxO = (tp.x - optTp.x) * 12;
        var dzO = (tp.z - optTp.z) * 12;
        var sepO = Math.sqrt(dxO*dxO + dzO*dzO);
        var delta = sepO - sepC;
        var deltaColor = Math.abs(delta) < 0.3 ? '#888' : (delta < 0 ? '#1D9E75' : '#ef4444');
        var deltaSign = delta > 0 ? '+' : '';
        lines.push(
          '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:6px;padding:2px 0;flex-wrap:wrap;">' +
          '<span style="color:' + t.color + ';font-weight:600;white-space:nowrap;">vs ' + t.label + '</span>' +
          '<span style="white-space:nowrap;"><span style="color:#3b82f6;">' + sepC.toFixed(1) + '"</span> \u2192 ' +
          '<span style="color:#f59e0b;">' + sepO.toFixed(1) + '"</span> ' +
          '<span style="color:' + deltaColor + ';font-size:9px;">(' + deltaSign + delta.toFixed(1) + '")</span></span>' +
          '</div>'
        );
      });
    }
    csLegend.innerHTML = lines.join('');
  }

  setup(); buildLegend(); draw();
  csSetup(); csDraw();
  var tRESIZE;
  window.addEventListener('resize', function(){
    clearTimeout(tRESIZE); tRESIZE = setTimeout(function(){
      setup(); draw();
      csSetup(); csDraw();
    }, 80);
  });
})();
</script>
"""

    js = js.replace('TRAJ_JSON', traj_json)
    js = js.replace('TUNNEL_Y', str(round(tunnel_y, 2)))
    js = js.replace('WORLD_Y_MAX', str(world_y_max))
    js = js.replace('WORLD_X_MIN', str(world_x_min))
    js = js.replace('WORLD_X_MAX', str(world_x_max))
    js = js.replace('WORLD_Z_MIN', str(world_z_min))
    js = js.replace('WORLD_Z_MAX', str(world_z_max))
    js = js.replace('TRAJID', selected_pt + '_traj')
    js = js.replace('tRESIZE', 't_' + selected_pt + '_resize')

    components.html(html + js, height=600, scrolling=False)


# ── Custom CSV upload tab ──────────────────────────────────────────────────
def render_pitcher_editor(pid, profile, existing_profiles, norm_tables,
                          stuff_models, sensitivity_radii):
    """
    Override panel for custom (synthetic) pitchers. Lets the user edit the
    parsed mechanics + current shape per pitch, then recomputes comps, bounds,
    and grades through the same pipeline as the original upload.
    Only shown for synthetic profiles.
    """
    if not profile.get('is_synthetic'):
        return

    with st.expander("✎ Edit pitcher (override parsed values)", expanded=False):
        st.markdown(
            '<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#8c857a;'
            'margin-bottom:0.6rem;">Override the values parsed from the CSV. Changing any '
            'value recomputes comps, achievable bounds, and grades. Leave values as-is to '
            'keep the parsed numbers.</div>',
            unsafe_allow_html=True
        )

        pitches = profile.get('pitches', {})
        with st.form(key=f"edit_{pid}"):
            edits = {}
            for pt, info in pitches.items():
                fx   = info.get('fixed', {})
                semi = info.get('semi_fixed', {})
                opt  = info.get('optimizable', {})

                st.markdown(
                    f'<div style="font-family:Barlow Condensed,sans-serif;font-size:0.95rem;'
                    f'font-weight:700;color:{pitch_color(pt)};text-transform:uppercase;'
                    f'margin:0.5rem 0 0.2rem;">{pitch_name(pt)}</div>',
                    unsafe_allow_html=True
                )
                c1, c2, c3 = st.columns(3)
                c4, c5, c6, c7 = st.columns(4)
                e = {}
                with c1:
                    e['velo'] = st.number_input(
                        "Velo (mph)", value=float(semi.get('velo_mean') or 0.0),
                        step=0.1, format="%.1f", key=f"{pid}_{pt}_velo")
                with c2:
                    e['pfx_x'] = st.number_input(
                        "H-Break (in)", value=float(opt.get('pfx_x_mean') or 0.0),
                        step=0.1, format="%.1f", key=f"{pid}_{pt}_px")
                with c3:
                    e['pfx_z'] = st.number_input(
                        "V-Break (in)", value=float(opt.get('pfx_z_mean') or 0.0),
                        step=0.1, format="%.1f", key=f"{pid}_{pt}_pz")
                with c4:
                    e['rpx'] = st.number_input(
                        "Arm slot rpx (ft)", value=float(fx.get('release_pos_x') or 0.0),
                        step=0.01, format="%.2f", key=f"{pid}_{pt}_rpx")
                with c5:
                    e['rpz'] = st.number_input(
                        "Arm slot rpz (ft)", value=float(fx.get('release_pos_z') or 0.0),
                        step=0.01, format="%.2f", key=f"{pid}_{pt}_rpz")
                with c6:
                    e['ext'] = st.number_input(
                        "Extension (ft)", value=float(fx.get('release_extension') or 0.0),
                        step=0.01, format="%.2f", key=f"{pid}_{pt}_ext")
                with c7:
                    e['spin_eff'] = st.number_input(
                        "Active spin (rpm)", value=float(fx.get('spin_efficiency') or 0.0),
                        step=10.0, format="%.0f", key=f"{pid}_{pt}_se")
                e['spin_axis'] = st.number_input(
                    "Spin axis (deg)", value=float(semi.get('spin_axis_mean') or 0.0),
                    step=1.0, format="%.0f", key=f"{pid}_{pt}_sa")
                edits[pt] = e

            colA, colB = st.columns([1, 3])
            with colA:
                submitted = st.form_submit_button("Apply & recompute")
            with colB:
                reset = st.form_submit_button("Reset to parsed values")

        if reset:
            # Drop edited copy so the original parsed profile is used again
            orig = st.session_state.get('synthetic_profiles_original', {}).get(pid)
            if orig is not None:
                import copy
                st.session_state['synthetic_profiles'][pid] = copy.deepcopy(orig)
                st.session_state['synthetic_comp_bounds'][pid] = \
                    copy.deepcopy(orig.get('_comp_bounds', {}))
                # Clear the number_input widget keys so they repopulate from the
                # restored profile values rather than the user's last typed entries
                for pt in pitches.keys():
                    for suffix in ('velo', 'px', 'pz', 'rpx', 'rpz', 'ext', 'se', 'sa'):
                        st.session_state.pop(f"{pid}_{pt}_{suffix}", None)
                st.rerun()

        if submitted:
            import copy
            # Preserve a pristine copy the first time we edit, for "reset"
            if 'synthetic_profiles_original' not in st.session_state:
                st.session_state['synthetic_profiles_original'] = {}
            if pid not in st.session_state['synthetic_profiles_original']:
                st.session_state['synthetic_profiles_original'][pid] = copy.deepcopy(profile)

            edited = copy.deepcopy(profile)
            for pt, e in edits.items():
                info = edited['pitches'][pt]
                fx   = info['fixed']

                # Detect whether the shape/velo/release actually changed — if so we
                # must recompute the raw trajectory frame (vx0..az) so the flight-path
                # and tunneling visualizations reflect the new shape. The stored
                # trajectory components came from the CSV and describe the ORIGINAL
                # pitch; leaving them stale makes the charts ignore edits.
                old_px = info['optimizable'].get('pfx_x_mean')
                old_pz = info['optimizable'].get('pfx_z_mean')
                old_velo = info['semi_fixed'].get('velo_mean')
                shape_changed = (
                    old_px != e['pfx_x'] or old_pz != e['pfx_z'] or old_velo != e['velo']
                    or fx.get('release_pos_x') != e['rpx']
                    or fx.get('release_pos_z') != e['rpz']
                    or fx.get('release_extension') != e['ext']
                )

                info['semi_fixed']['velo_mean']      = e['velo']
                info['semi_fixed']['velo_lo']        = round(e['velo'] - 1.5, 1)
                info['semi_fixed']['velo_hi']        = round(e['velo'] + 1.5, 1)
                info['semi_fixed']['spin_axis_mean'] = e['spin_axis']
                info['optimizable']['pfx_x_mean']    = e['pfx_x']
                info['optimizable']['pfx_z_mean']    = e['pfx_z']
                fx['release_pos_x']     = e['rpx']
                fx['release_pos_z']     = e['rpz']
                fx['release_extension'] = e['ext']
                fx['release_pos_y']     = 60.5 - e['ext']
                fx['spin_efficiency']   = e['spin_eff']

                # Recompute the trajectory frame to match the edited shape, preserving
                # the original release direction (vy0 sign etc.) where we have it.
                if shape_changed and fx.get('vy0') is not None:
                    from tunneling import make_pitch_dict_from_optimized
                    td = make_pitch_dict_from_optimized(
                        e['pfx_x'], e['pfx_z'], e['velo'],
                        e['rpx'], 60.5 - e['ext'], e['rpz'],
                        vx0_actual=fx.get('vx0'), vy0_actual=fx.get('vy0'),
                        vz0_actual=fx.get('vz0'), ay_actual=fx.get('ay'),
                    )
                    fx['vx0'] = td['vx0']; fx['vy0'] = td['vy0']; fx['vz0'] = td['vz0']
                    fx['ax']  = td['ax'];  fx['ay']  = td['ay'];  fx['az']  = td['az']

            with st.spinner("Recomputing comps, bounds, and grades..."):
                import synthetic_comps
                edited, comp_bounds = synthetic_comps.recompute_synthetic_profile(
                    edited, existing_profiles, sensitivity_radii,
                    norm_tables, stuff_models, predict_grades, predict_weighted_grades
                )

            st.session_state['synthetic_profiles'][pid] = edited
            st.session_state['synthetic_comp_bounds'][pid] = comp_bounds
            st.success("Updated — values recomputed.")
            st.rerun()


def render_upload_tab(existing_profiles, norm_tables, stuff_models, sensitivity_radii):
    import csv_import
    import synthetic_comps

    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#8c857a;margin:0.5rem 0 1rem;">'
        'Upload a Trackman or Rapsodo CSV to analyze a pitcher without Statcast data. '
        'The tool builds a profile from mechanics + velocity and recommends optimal shapes. '
        'Stuff+ grades are relative to MLB averages.</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload pitch CSV", type=['csv'],
                                label_visibility="collapsed")
    col_a, col_b = st.columns(2)
    with col_a:
        name_override = st.text_input("Pitcher name (optional)",
                                      placeholder="leave blank to use CSV name")

    if uploaded is None:
        return

    file_bytes = uploaded.getvalue()

    # Only parse + store ONCE per uploaded file. The file_uploader widget retains
    # the file across reruns, so without this guard every rerun (including the one
    # triggered by the Edit-pitcher recompute) would re-parse the CSV and overwrite
    # the user's edits in session state with the original parsed values.
    import hashlib
    file_sig = hashlib.md5(file_bytes).hexdigest() + "|" + (name_override.strip() or "")
    already_loaded = st.session_state.get('_loaded_csv_sig') == file_sig

    if already_loaded:
        # File already processed this session — show a brief confirmation and the
        # roster, but DON'T re-parse (which would wipe edits).
        loaded_profiles = st.session_state.get('synthetic_profiles', {})
        if loaded_profiles:
            st.caption("CSV already loaded. Edits are preserved — switch to the "
                       "Search tab to analyze, or upload a different file to reload.")
            for sid, syn in loaded_profiles.items():
                pts = ', '.join(f"{pitch_name(pt)} ({info.get('n_pitches', info.get('n','?'))})"
                                for pt, info in syn['pitches'].items())
                st.markdown(
                    f'<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#8c857a;">'
                    f'<b style="color:#3a352c;">{syn["player_name"]}</b> '
                    f'({syn["p_throws"]}HP) — {pts}</div>',
                    unsafe_allow_html=True
                )
        return

    with st.spinner("Parsing CSV and finding comps..."):
        syn_profiles, messages = csv_import.parse_csv(
            file_bytes, pitcher_name_override=name_override.strip() or None
        )

    # Show parse messages
    for m in messages:
        if 'Missing' in m or 'No ' in m or 'Could not' in m:
            st.warning(m)
        else:
            st.caption(m)

    if not syn_profiles:
        return

    # Find comps + finalize grades for each synthetic profile
    with st.spinner("Computing comps and grades..."):
        finalized = {}
        for sid, syn in syn_profiles.items():
            syn, comp_bounds = synthetic_comps.recompute_synthetic_profile(
                syn, existing_profiles, sensitivity_radii,
                norm_tables, stuff_models, predict_grades, predict_weighted_grades
            )
            finalized[sid] = syn

    # Persist in session state
    if 'synthetic_profiles' not in st.session_state:
        st.session_state['synthetic_profiles'] = {}
    if 'synthetic_comp_bounds' not in st.session_state:
        st.session_state['synthetic_comp_bounds'] = {}
    if 'synthetic_profiles_original' not in st.session_state:
        st.session_state['synthetic_profiles_original'] = {}
    import copy as _copy
    for sid, syn in finalized.items():
        st.session_state['synthetic_profiles'][sid] = syn
        st.session_state['synthetic_comp_bounds'][sid] = syn.get('_comp_bounds', {})
        # Keep a pristine copy so the editor's "reset to parsed values" works
        st.session_state['synthetic_profiles_original'][sid] = _copy.deepcopy(syn)

    # Mark this file as loaded so we don't re-parse (and clobber edits) on rerun
    st.session_state['_loaded_csv_sig'] = file_sig

    # Rerun so the search tab's dropdown rebuilds with the just-uploaded pitcher
    # included. Without this, the upload is stored in session state but the search
    # options were already built earlier in this same run, so the new pitcher
    # wouldn't appear until some other interaction forced a rerun.
    st.rerun()

    # Summary of what was loaded
    st.success(f"Loaded {len(finalized)} pitcher(s). Switch to the Search tab — "
               f"they'll appear at the top of the dropdown, tagged \"(uploaded)\".")
    for sid, syn in finalized.items():
        pts = ', '.join(f"{pitch_name(pt)} ({info['n_pitches']})"
                        for pt, info in syn['pitches'].items())
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#8c857a;">'
            f'<b style="color:#3a352c;">{syn["player_name"]}</b> ({syn["p_throws"]}HP) — {pts}</div>',
            unsafe_allow_html=True
        )


def main():
    data = load_data()
    if not data.get('profiles'):
        st.error("Could not load pitcher profiles. Check data/ folder.")
        return

    profiles     = data['profiles']
    stuff_models = data['stuff_models'] or {}
    norm_tables  = data['norm_tables']  or {}
    feat_imp     = data['feature_importance'] or {}

    # Merge any uploaded synthetic profiles from session state
    if 'synthetic_profiles' in st.session_state and st.session_state['synthetic_profiles']:
        profiles = {**profiles, **st.session_state['synthetic_profiles']}

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-family:Barlow Condensed,sans-serif;font-size:0.65rem;font-weight:700;
                  color:#8c857a;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:4px;">
        Pitch Scout
      </div>
      <div style="font-family:Barlow Condensed,sans-serif;font-size:2.4rem;font-weight:900;
                  color:#3a352c;line-height:1;letter-spacing:0.02em;">
        Shape Optimizer
      </div>
      <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#8c857a;margin-top:4px;">
        Find the optimal movement profile for each pitch given fixed mechanics
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pitcher search / upload tabs ──────────────────────────────────────
    tab_search, tab_upload = st.tabs(["Search MLB pitchers", "Upload custom data"])

    with tab_upload:
        render_upload_tab(profiles, norm_tables, stuff_models,
                          data.get('sensitivity_radii'))

    with tab_search:
        # Build a searchable dropdown of all pitchers, displayed as "First Last".
        # Streamlit's selectbox filters the list live as you type (matches anywhere
        # in the label, so first name, last name, or full name all work), and you can
        # either press enter on the highlighted match or click a name from the list.
        # Uploaded (synthetic) pitchers are included too, tagged "(uploaded)" so they
        # stay reachable and are distinguishable from MLB pitchers.
        # Re-merge synthetic profiles here from the latest session state. The merge
        # at the top of main() runs BEFORE render_upload_tab() stores a new upload,
        # so without this a just-uploaded pitcher wouldn't appear in the dropdown
        # until an extra rerun. Reading session state again here picks it up now.
        if st.session_state.get('synthetic_profiles'):
            profiles = {**profiles, **st.session_state['synthetic_profiles']}
        label_to_pid = {}
        upload_labels = []
        mlb_labels = []
        for pid, p in profiles.items():
            base = to_first_last(p.get('player_name', ''))
            if not base:
                continue
            if p.get('is_synthetic'):
                label = f"{base} (uploaded)"
                upload_labels.append(label)
            else:
                label = base
                mlb_labels.append(label)
            label_to_pid[label] = pid
        # Uploaded pitchers first (most likely what the user wants right after upload),
        # then MLB pitchers alphabetically.
        sorted_labels = sorted(upload_labels) + sorted(mlb_labels)

        chosen_label = st.selectbox(
            "Search pitcher",
            options=sorted_labels,
            index=None,
            placeholder="Search a pitcher by name… (e.g. Bryan Woo)",
            label_visibility="collapsed",
        )

    if not chosen_label:
        st.markdown(
            '<div style="font-family:DM Mono,monospace;font-size:0.75rem;color:#8c857a;'
            'margin-top:2rem;text-align:center;">Search for a pitcher or upload custom data to begin</div>',
            unsafe_allow_html=True
        )
        return

    pid = label_to_pid[chosen_label]
    profile = profiles[pid]

    # For synthetic (uploaded) pitchers, always read the freshest copy from
    # session state so an edit made this session is authoritative even if the
    # merged `profiles` dict above captured a stale reference.
    if profile.get('is_synthetic'):
        _sp = st.session_state.get('synthetic_profiles', {})
        if pid in _sp:
            profile = _sp[pid]

    pitcher_name = to_first_last(profile['player_name'])
    p_throws     = profile.get('p_throws', 'R')
    pitches      = profile.get('pitches', {})

    if not pitches:
        st.warning("No pitch data available for this pitcher.")
        return

    # ── Arsenal overview ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:Barlow Condensed,sans-serif;font-size:1.5rem;font-weight:800;'
        f'color:#3a352c;margin:1.2rem 0 0.4rem;">{pitcher_name}'
        f'<span style="font-size:0.9rem;font-weight:400;color:#8c857a;margin-left:10px;">'
        f'{p_throws}HP</span></div>',
        unsafe_allow_html=True
    )

    # Precompute which pitches improve both Stuff+ and tunneling when optimized
    with st.spinner("Analyzing arsenal..."):
        arsenal_improvements = compute_arsenal_improvements(profile, norm_tables, stuff_models)

    # Arsenal grade cards — show combined + per-handedness grades
    n_pitches = len(pitches)
    cols      = st.columns(n_pitches)
    for col, (pt, info) in zip(cols, pitches.items()):
        g    = info.get('grades', {})
        gr   = info.get('grades_rhh', {})
        gl   = info.get('grades_lhh', {})
        sp   = g.get('stuff_plus', '—')
        ap   = g.get('arsenal_plus', '—')
        cp   = g.get('contact_plus', '—')
        sp_r = gr.get('stuff_plus', '—')
        sp_l = gl.get('stuff_plus', '—')
        pc   = pitch_color(pt)
        # Check if this pitch has dual-improvement potential
        improvement = arsenal_improvements.get(pt, {})
        dual_better = improvement.get("improves_both", False)
        improved_pairs = improvement.get("improved_pairs", [])
        badge_html = ""
        shadow_css = ""
        if dual_better:
            pair_text = " + ".join(pitch_name(p) for p in improved_pairs[:2])
            badge_html = (
                '<div style="display:inline-block;background:rgba(16,185,129,0.18);'
                'border:1px solid rgba(16,185,129,0.4);border-radius:3px;padding:1px 6px;'
                'margin-left:6px;font-size:0.5rem;font-weight:700;color:#1D9E75;'
                'text-transform:uppercase;letter-spacing:0.04em;">Stuff+ ↑ tun ↑</div>'
            )
            shadow_css = 'box-shadow:0 0 0 1.5px rgba(16,185,129,0.5);'
        col.markdown(
            f'<div style="background:#faf8f4;border:1px solid rgba(255,255,255,0.06);'
            f'border-top:3px solid {pc};border-radius:0 0 8px 8px;padding:0.8rem 0.9rem;{shadow_css}">'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:0.75rem;'
            f'font-weight:700;color:{pc};text-transform:uppercase;margin-bottom:0.5rem;'
            f'display:flex;align-items:center;">'
            f'<span>{pitch_name(pt)}</span>{badge_html}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-end;">'
            f'<div>'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:1.8rem;'
            f'font-weight:900;color:{grade_color(sp) if isinstance(sp,float) else "#3a352c"};'
            f'line-height:1;">{sp if isinstance(sp, float) else "—"}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.55rem;color:#8c857a;'
            f'text-transform:uppercase;letter-spacing:0.08em;">overall Stuff+</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#b0a99c;margin-top:4px;">'
            f'vs R: {sp_r if isinstance(sp_r, float) else "—"} '
            f'&nbsp;|&nbsp; vs L: {sp_l if isinstance(sp_l, float) else "—"}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8c857a;">Ars+ {ap}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8c857a;">Con+ {cp}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#b0a99c;">n={info["n"]:,}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

    # ── Edit pitcher (custom uploads only) ────────────────────────────────
    render_pitcher_editor(pid, profile, profiles, norm_tables, stuff_models,
                          data.get('sensitivity_radii'))

    # ── Pitch selector ────────────────────────────────────────────────────
    pt_options     = list(pitches.keys())
    pt_labels      = [f"{pitch_name(pt)} ({pt})" for pt in pt_options]
    selected_label = st.selectbox("Select pitch to optimize", pt_labels,
                                  key=f"pitchsel_{pid}")
    selected_pt    = pt_options[pt_labels.index(selected_label)]
    pt_info        = pitches[selected_pt]

    # Check model exists for at least one matchup
    has_model = any((p_throws, s, selected_pt) in stuff_models for s in ['R', 'L'])
    if not has_model:
        st.warning(f"No stuff model available for {pitch_name(selected_pt)}.")
        return

    pc = pitch_color(selected_pt)

    # Attach pitcher-level info needed by predict_grades
    pt_info['p_throws']         = p_throws
    pt_info['primary_fb_velo']  = profile.get('primary_fb_velo')
    pt_info['primary_fb_pfx_x'] = profile.get('primary_fb_pfx_x')
    pt_info['primary_fb_pfx_z'] = profile.get('primary_fb_pfx_z')

    opt        = pt_info['optimizable']
    semi       = pt_info['semi_fixed']
    curr_pfx_x = opt['pfx_x_mean']
    curr_pfx_z = opt['pfx_z_mean']
    curr_velo  = semi['velo_mean']

    # ── Run optimizer ─────────────────────────────────────────────────────
    with st.spinner(f"Optimizing {pitch_name(selected_pt)} shape..."):
        curr_spin_eff = pt_info['fixed'].get('spin_efficiency')
        rhh_w = norm_tables.get('rhh_weight', 0.58)
        lhh_w = norm_tables.get('lhh_weight', 0.42)

        # Run both handedness optimizations
        bx_r, bz_r, bse_r, best_r = run_optimizer(
            selected_pt, pt_info, norm_tables, stuff_models, stand='R')
        bx_l, bz_l, bse_l, best_l = run_optimizer(
            selected_pt, pt_info, norm_tables, stuff_models, stand='L')

        # Current grades per handedness
        curr_r = pt_info.get('grades_rhh', {})
        curr_l = pt_info.get('grades_lhh', {})

        # Combined weighted optimal — find the shape that maximizes
        # the weighted average Stuff+ across both handedness matchups
        def combined_rv(pfx_x, pfx_z):
            p_throws = pt_info.get('p_throws', 'R')
            model_r  = PitchScoutStuffModel(stuff_models, norm_tables, p_throws, 'R', selected_pt)
            model_l  = PitchScoutStuffModel(stuff_models, norm_tables, p_throws, 'L', selected_pt)
            feat_r   = build_features(selected_pt, pfx_x, pfx_z, curr_spin_eff, curr_velo, pt_info)
            feat_l   = build_features(selected_pt, pfx_x, pfx_z, curr_spin_eff, curr_velo, pt_info)
            rv_r     = model_r.predict_rv(feat_r)
            rv_l     = model_l.predict_rv(feat_l)
            return rv_r * rhh_w + rv_l * lhh_w

        # Coarse grid on combined objective
        opt_bounds = pt_info['optimizable']
        x_vals     = np.linspace(opt_bounds['pfx_x_lo'], opt_bounds['pfx_x_hi'], 20)
        z_vals     = np.linspace(opt_bounds['pfx_z_lo'], opt_bounds['pfx_z_hi'], 20)
        best_rv_c  = float('inf')
        best_x0_c  = [curr_pfx_x, curr_pfx_z]
        for px in x_vals:
            for pz in z_vals:
                rv = combined_rv(px, pz)
                if rv < best_rv_c:
                    best_rv_c = rv
                    best_x0_c = [px, pz]

        # Scipy refinement on combined objective
        try:
            res = minimize(
                lambda x: combined_rv(x[0], x[1]),
                x0=best_x0_c,
                bounds=[(opt_bounds['pfx_x_lo'], opt_bounds['pfx_x_hi']),
                        (opt_bounds['pfx_z_lo'], opt_bounds['pfx_z_hi'])],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-9},
            )
            bx_c, bz_c = res.x
        except Exception:
            bx_c, bz_c = best_x0_c

        curr_c = pt_info.get('grades', {})
        best_c, _, _ = predict_weighted_grades(
            selected_pt, float(bx_c), float(bz_c), curr_spin_eff,
            curr_velo, pt_info, norm_tables, stuff_models)

        # Safeguard: the optimizer minimizes weighted RV, but display uses weighted
        # Stuff+, which is a nonlinear function of per-handedness RV. In rare cases the
        # min-RV shape can grade slightly below the current shape on weighted Stuff+.
        # The current shape is always achievable, so if optimization doesn't beat it,
        # fall back to the current shape (no recommendation to change).
        curr_sp = curr_c.get('stuff_plus')
        opt_sp  = best_c.get('stuff_plus')
        if curr_sp is not None and opt_sp is not None and opt_sp < curr_sp:
            bx_c, bz_c = curr_pfx_x, curr_pfx_z
            best_c = dict(curr_c)

    # ── Section header ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:0.5rem 0 0.8rem;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{pc};display:inline-block;"></span>'
        f'<span style="font-family:Barlow Condensed,sans-serif;font-size:1.2rem;font-weight:800;'
        f'color:{pc};text-transform:uppercase;">{pitch_name(selected_pt)}</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#8c857a;">— shape optimization</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;color:#b0a99c;margin-left:auto;">'
        f'weighted {rhh_w:.0%} RHH / {lhh_w:.0%} LHH · comp-derived bounds</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Compute VAA (current measured + optimized derived) for FF/SI display ──
    _cur_vaa = None
    _opt_vaa = None
    if selected_pt in VAA_PITCH_TYPES:
        _fx = pt_info.get('fixed', {})
        _measured_vaa = _fx.get('vaa_mean')
        if _measured_vaa is not None:
            _cur_vaa = _measured_vaa
            try:
                from tunneling import optimized_vaa
                _rpy = _fx.get('release_pos_y') or (60.5 - (_fx.get('release_extension') or 6.0))
                _opt_vaa = optimized_vaa(
                    _measured_vaa, curr_pfx_x, curr_pfx_z,
                    float(bx_c), float(bz_c), curr_velo,
                    _fx.get('release_pos_x', 0.0) or 0.0, _rpy,
                    _fx.get('release_pos_z', 6.0) or 6.0,
                    vx0_actual=_fx.get('vx0'), vy0_actual=_fx.get('vy0'),
                    vz0_actual=_fx.get('vz0'), ay_actual=_fx.get('ay'),
                )
            except Exception:
                _opt_vaa = None

    # ── Primary card — combined optimal ───────────────────────────────────
    render_shape_card(
        "overall — optimized (weighted vs RHH + LHH)",
        selected_pt, curr_c, best_c,
        curr_pfx_x, curr_pfx_z,
        float(bx_c), float(bz_c), curr_velo, pc,
        current_vaa=_cur_vaa, opt_vaa=_opt_vaa
    )

    # ── Build arsenal data for chart visualization ─────────────────────────
    # Check synthetic comp bounds first, then fall back to pre-computed MLB comps
    _syn_cb = st.session_state.get('synthetic_comp_bounds', {})
    if int(pid) in _syn_cb:
        _comp_bounds = _syn_cb[int(pid)].get(selected_pt, {})
    else:
        _comp_bounds = (data.get('comp_profiles') or {}).get(int(pid), {}).get(selected_pt, {})

    # Identify which other pitches form valid tunnel pairs with the selected one
    _arsenal_list = []
    _tunnel_dict = {}
    from tunneling import compute_tunnel_metrics, make_pitch_dict_from_optimized
    traj_keys = ['release_pos_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    sel_fix = pt_info.get('fixed', {})
    if all(sel_fix.get(k) is not None for k in traj_keys):
        sel_current = {
            'release_pos_x': sel_fix['release_pos_x'],
            'release_pos_y': sel_fix['release_pos_y'],
            'release_pos_z': sel_fix['release_pos_z'],
            'vx0': sel_fix['vx0'], 'vy0': sel_fix['vy0'], 'vz0': sel_fix['vz0'],
            'ax':  sel_fix['ax'],  'ay':  sel_fix['ay'],  'az':  sel_fix['az'],
        }
        sel_optimized = make_pitch_dict_from_optimized(
            float(bx_c), float(bz_c), curr_velo,
            sel_fix['release_pos_x'], sel_fix['release_pos_y'], sel_fix['release_pos_z'],
            vx0_actual=sel_fix['vx0'], vy0_actual=sel_fix['vy0'],
            vz0_actual=sel_fix['vz0'], ay_actual=sel_fix['ay'],
        )
        for other_pt, other_info in profile['pitches'].items():
            if other_pt == selected_pt: continue
            ofx = other_info.get('fixed', {})
            opt = other_info.get('optimizable', {})
            if opt.get('pfx_x_mean') is None or opt.get('pfx_z_mean') is None: continue
            is_pair = is_tunnel_pair(pt_info, other_info)
            _arsenal_list.append({
                'pt':      other_pt,
                'name':    pitch_name(other_pt),
                'pfx_x':   round(float(opt['pfx_x_mean']), 1),
                'pfx_z':   round(float(opt['pfx_z_mean']), 1),
                'color':   pitch_color(other_pt),
                'is_pair': is_pair,
            })
            # Compute tunnel metrics if both have trajectory data
            if all(ofx.get(k) is not None for k in traj_keys):
                other = {
                    'release_pos_x': ofx['release_pos_x'],
                    'release_pos_y': ofx['release_pos_y'],
                    'release_pos_z': ofx['release_pos_z'],
                    'vx0': ofx['vx0'], 'vy0': ofx['vy0'], 'vz0': ofx['vz0'],
                    'ax':  ofx['ax'],  'ay':  ofx['ay'],  'az':  ofx['az'],
                }
                cm = compute_tunnel_metrics(sel_current, other)
                om = compute_tunnel_metrics(sel_optimized, other)
                if cm and om:
                    _tunnel_dict[other_pt] = {
                        'curr_lbr':   cm['late_break_ratio'],
                        'opt_lbr':    om['late_break_ratio'],
                        'curr_tun':   cm['tunnel_sep_in'],
                        'opt_tun':    om['tunnel_sep_in'],
                        'curr_plate': cm['plate_sep_in'],
                        'opt_plate':  om['plate_sep_in'],
                    }

    render_movement_chart(
        selected_pt, curr_pfx_x, curr_pfx_z,
        float(bx_c), float(bz_c),
        _comp_bounds, pc, pitcher_name,
        arsenal_pitches=_arsenal_list,
        arsenal_tunnel_data=_tunnel_dict
    )

    # ── Tunneling analysis ───────────────────────────────────────────────────────────
    render_tunneling_section(selected_pt, pt_info, profile,
                              float(bx_c), float(bz_c), curr_velo)

    # ── 3D trajectory visualization ───────────────────────────────────────────────────
    with st.expander("📊 View flight paths", expanded=False):
        render_trajectory_chart(selected_pt, pt_info, profile,
                                float(bx_c), float(bz_c), curr_velo)

    with st.expander("🔍 View by handedness", expanded=False):
        col_r, col_l = st.columns(2)
        with col_r:
            render_shape_card(
                "vs RHH — optimized", selected_pt,
                curr_r, best_r, curr_pfx_x, curr_pfx_z,
                bx_r, bz_r, curr_velo, pc
            )
        with col_l:
            render_shape_card(
                "vs LHH — optimized", selected_pt,
                curr_l, best_l, curr_pfx_x, curr_pfx_z,
                bx_l, bz_l, curr_velo, pc
            )

    # ── Feature importance ────────────────────────────────────────────────
    if selected_pt in feat_imp:
        with st.expander("📊 What drives this pitch's grade", expanded=False):
            imp = feat_imp[selected_pt]
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#8c857a;'
                f'margin-bottom:0.75rem;">Feature importance — RV model '
                f'({pitch_name(selected_pt)}, averaged across matchups)</div>',
                unsafe_allow_html=True
            )
            FEAT_LABELS = {
                'release_speed':       'velocity',
                'pfx_x_in':            'h-break',
                'pfx_z_in':            'v-break',
                'release_pos_x':       'arm slot (h)',
                'release_pos_z':       'arm slot (v)',
                'release_extension':   'extension',
                'velo_diff_fb':        'velo diff vs FB',
                'pfx_x_diff_fb':       'h-break diff vs FB',
                'pfx_z_diff_fb':       'v-break diff vs FB',
                'spin_axis':           'spin axis',
                'spin_efficiency_raw': 'active spin',
                'ssw_interaction':     'SSW proxy',
                'vaa':                 'VAA',
            }
            for feat, pct in imp[:10]:
                label = FEAT_LABELS.get(feat, feat.replace('_', ' '))
                bar_w = int(pct * 2.5)
                col_feat, col_bar, col_pct = st.columns([2, 4, 1])
                col_feat.markdown(
                    f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;'
                    f'color:#8c857a;">{label}</span>', unsafe_allow_html=True)
                col_bar.markdown(
                    f'<div style="background:#e8e3d9;border-radius:3px;height:6px;margin-top:6px;">'
                    f'<div style="width:{min(bar_w,100)}%;height:6px;border-radius:3px;'
                    f'background:{pc};"></div></div>', unsafe_allow_html=True)
                col_pct.markdown(
                    f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;'
                    f'color:#8c857a;">{pct:.1f}%</span>', unsafe_allow_html=True)

    # ── Optimization bounds ───────────────────────────────────────────────
    with st.expander("⚙️ Optimization bounds", expanded=False):
        fixed = pt_info['fixed']
        semi  = pt_info['semi_fixed']
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#8c857a;'
            f'margin-bottom:0.6rem;">Fixed and semi-fixed attributes — '
            f'{pitcher_name} {pitch_name(selected_pt)}</div>',
            unsafe_allow_html=True
        )
        rows = [
            ('H-Break range',   f"{opt['pfx_x_lo']:.1f}\" → {opt['pfx_x_hi']:.1f}\"", 'optimizable'),
            ('V-Break range',   f"{opt['pfx_z_lo']:.1f}\" → {opt['pfx_z_hi']:.1f}\"", 'optimizable'),
            ('Velo range',      f"{semi['velo_lo']} → {semi['velo_hi']} mph",           'semi-fixed'),
            ('Spin axis range', f"{semi.get('spin_axis_lo','—')} → {semi.get('spin_axis_hi','—')}°", 'semi-fixed'),
            ('Arm slot (rpx)',  f"{fixed.get('release_pos_x', 0):.2f} ft",              'fixed'),
            ('Arm slot (rpz)',  f"{fixed.get('release_pos_z', 6):.2f} ft",              'fixed'),
            ('Extension',       f"{fixed.get('release_extension', 6):.2f} ft",          'fixed'),
            ('Active spin',     f"{fixed.get('spin_efficiency'):.0f} rpm" if fixed.get('spin_efficiency') else '—', 'fixed'),
        ]
        cat_colors = {'fixed': '#ef4444', 'semi-fixed': '#f59e0b', 'optimizable': '#1D9E75'}
        for label, val, cat in rows:
            cc = cat_colors[cat]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8c857a;">{label}</span>'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;color:#3a352c;">{val}</span>'
                f'<span style="font-family:Barlow Condensed,sans-serif;font-size:0.58rem;font-weight:700;'
                f'color:{cc};text-transform:uppercase;letter-spacing:0.08em;">{cat}</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()
