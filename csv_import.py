"""
csv_import.py
=============
Parse Trackman / Rapsodo pitch CSVs into Pitch Scout synthetic pitcher profiles.

Handles coordinate system conversion (Trackman x-toward-3B → Statcast x-toward-1B),
unit normalization, pitch type mapping, outlier filtering, and trajectory feature
extraction (or derivation where not measured).

The output profile matches the structure of pitcher_stuff_profiles.pkl entries so
the existing optimizer, chart, and tunneling code work unchanged.
"""

import pandas as pd
import numpy as np
import io


# Pitch type name → our short codes
PITCH_TYPE_MAP = {
    'fastball': 'FF', 'four-seam': 'FF', 'fourseam': 'FF', '4-seam': 'FF',
    'four-seam fastball': 'FF', '4-seam fastball': 'FF', 'ff': 'FF',
    'sinker': 'SI', 'two-seam': 'SI', 'twoseam': 'SI', '2-seam': 'SI',
    'two-seam fastball': 'SI', 'si': 'SI', 'ft': 'SI',
    'cutter': 'FC', 'fc': 'FC',
    'slider': 'SL', 'sl': 'SL',
    'curveball': 'CU', 'curve': 'CU', 'cu': 'CU', 'kc': 'CU',
    'changeup': 'CH', 'change-up': 'CH', 'change up': 'CH', 'ch': 'CH',
    'splitter': 'FS', 'split-finger': 'FS', 'fs': 'FS', 'split': 'FS',
    'sweeper': 'ST', 'st': 'ST',
    'slurve': 'SV', 'sv': 'SV',
}

VALID_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'FS', 'ST', 'SV']
FASTBALL_TYPES = {'FF', 'SI', 'FC'}

MIN_PITCHES_PER_TYPE = 5  # low bar — shape rec only needs stable mechanics


# Column name candidates for each canonical field (lowercased for matching)
TRACKMAN_COLUMNS = {
    'pitch_type':   ['taggedpitchtype', 'autopitchtype', 'pitchtype'],
    'pitcher':      ['pitcher', 'pitcherid', 'pitchername'],
    'p_throws':     ['pitcherthrows', 'throws'],
    'batter_side':  ['batterside', 'batterhand'],
    'rel_speed':    ['relspeed', 'velocity', 'velo'],
    'rel_height':   ['relheight', 'releaseheight'],
    'rel_side':     ['relside', 'releaseside'],
    'extension':    ['extension', 'releaseextension'],
    'spin_rate':    ['spinrate', 'totalspin'],
    'spin_axis':    ['spinaxis', 'spindirection'],
    'ivb':          ['inducedvertbreak', 'ivb', 'verticalbreakinduced'],
    'hb':           ['horzbreak', 'hb', 'horizontalbreak'],
    'vaa':          ['vertapprangle', 'vaa', 'verticalapproachangle'],
    'vx0':          ['vx0'],
    'vy0':          ['vy0'],
    'vz0':          ['vz0'],
    'ax0':          ['ax0', 'ax'],
    'ay0':          ['ay0', 'ay'],
    'az0':          ['az0', 'az'],
    'x0':           ['x0'],
    'y0':           ['y0'],
    'z0':           ['z0'],
    'spin_eff':     ['spinefficiency', 'efficiency', 'activespinpct'],
    'true_spin':    ['truespin', 'activespin', 'truespin(rpm)'],
}


def _find_column(df_cols_lower, candidates):
    """Return the actual column name matching any candidate, or None."""
    for cand in candidates:
        if cand in df_cols_lower:
            return df_cols_lower[cand]
    return None


def detect_format(df):
    """Guess whether this is Trackman, Rapsodo, or unknown based on columns."""
    cols_lower = set(c.lower().strip() for c in df.columns)
    if 'relspeed' in cols_lower or 'inducedvertbreak' in cols_lower:
        return 'trackman'
    if 'spin efficiency' in cols_lower or 'true spin (rpm)' in cols_lower or 'gyro degree' in cols_lower:
        return 'rapsodo'
    # Generic fallback
    return 'unknown'


def _map_columns(df):
    """Build a mapping from canonical field → actual column name in df."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canonical, candidates in TRACKMAN_COLUMNS.items():
        col = _find_column(df_cols_lower, candidates)
        if col:
            mapping[canonical] = col
    return mapping


def _normalize_pitch_type(raw):
    if pd.isna(raw):
        return None
    key = str(raw).lower().strip()
    return PITCH_TYPE_MAP.get(key)


def _normalize_throws(raw):
    if pd.isna(raw):
        return 'R'
    s = str(raw).strip().upper()
    if s.startswith('L'):
        return 'L'
    return 'R'


def parse_csv(file_bytes, pitcher_name_override=None, skiprows=0):
    """
    Parse a CSV file (bytes) into a list of synthetic pitcher profiles.

    Returns: (profiles_dict, messages)
      profiles_dict: {synthetic_id: profile} matching pitcher_stuff_profiles structure
      messages: list of info/warning strings about the parse
    """
    messages = []

    # Read CSV — try with skiprows for Rapsodo (4 header rows) if first attempt looks wrong
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), skiprows=skiprows)
    except Exception as e:
        return {}, [f"Could not read CSV: {e}"]

    fmt = detect_format(df)
    messages.append(f"Detected format: {fmt}")

    # If unknown and we haven't skipped rows, Rapsodo may have metadata header — retry
    if fmt == 'unknown' and skiprows == 0:
        try:
            df2 = pd.read_csv(io.BytesIO(file_bytes), skiprows=4)
            if detect_format(df2) != 'unknown':
                df = df2
                fmt = detect_format(df)
                messages.append(f"Retried with skiprows=4, detected: {fmt}")
        except Exception:
            pass

    mapping = _map_columns(df)

    # Required fields for a shape recommendation
    required = ['rel_speed', 'rel_height', 'rel_side', 'extension', 'ivb', 'hb', 'pitch_type']
    missing = [r for r in required if r not in mapping]
    if missing:
        return {}, messages + [f"Missing required columns: {missing}. "
                               f"Found columns: {list(df.columns)[:20]}"]

    # Determine if trajectory data is present (for accurate tunneling)
    has_traj = all(k in mapping for k in ['vx0', 'vy0', 'vz0', 'ax0', 'ay0', 'az0'])
    if has_traj:
        messages.append("Trajectory data present — tunneling will use measured values.")
    else:
        messages.append("No trajectory data — tunneling will use Magnus-derived estimates.")

    # Group by pitcher
    pitcher_col = mapping.get('pitcher')
    if pitcher_col is None:
        # Single pitcher assumed
        df['_pitcher_grp'] = pitcher_name_override or 'Uploaded Pitcher'
        pitcher_col = '_pitcher_grp'

    profiles = {}
    synthetic_id = 900000  # high IDs to avoid collision with MLB ids

    for pitcher_val, pdf in df.groupby(pitcher_col):
        # Resolve name and handedness
        if pitcher_name_override and df[pitcher_col].nunique() == 1:
            name = pitcher_name_override
        else:
            name = str(pitcher_val)
        p_throws = _normalize_throws(
            pdf[mapping['p_throws']].mode().iloc[0] if 'p_throws' in mapping and len(pdf) else 'R'
        )

        pitches = {}
        for raw_pt, ptdf in pdf.groupby(mapping['pitch_type']):
            pt = _normalize_pitch_type(raw_pt)
            if pt is None or pt not in VALID_PITCH_TYPES:
                continue

            # Clean: drop rows missing core fields
            core = [mapping['rel_speed'], mapping['hb'], mapping['ivb']]
            clean = ptdf.dropna(subset=core).copy()
            if len(clean) < MIN_PITCHES_PER_TYPE:
                continue

            # Outlier filter — drop pitches >2.5 MAD from median on movement
            for col in [mapping['hb'], mapping['ivb'], mapping['rel_speed']]:
                med = clean[col].median()
                mad = (clean[col] - med).abs().median()
                if mad > 0:
                    clean = clean[(clean[col] - med).abs() <= 3.5 * mad]
            if len(clean) < MIN_PITCHES_PER_TYPE:
                clean = ptdf.dropna(subset=core).copy()  # revert if filter too aggressive

            # COORDINATE CONVERSION — careful, Trackman mixes two conventions:
            #   * RelSide / HorzBreak use x-positive-toward-3B  -> FLIP to Statcast
            #   * The trajectory frame (x0,y0,z0,vx0..az0) is ALREADY in Statcast
            #     convention (x-positive-toward-1B). Verified across all pitchers:
            #     RHP x0 ~ -1.3 (negative/3B side, matches Statcast), vx0 ~ +5
            #     (positive, matches Statcast). So those must NOT be flipped.
            # Flipping the trajectory frame (the old bug) made vx0/ax contradict the
            # release position and the flight path streaked sideways in the 3D view.
            hb_mean  = -float(clean[mapping['hb']].mean())   # flip: Trackman 3B+ -> Statcast
            ivb_mean =  float(clean[mapping['ivb']].mean())  # vertical, unchanged
            velo     =  float(clean[mapping['rel_speed']].mean())
            rpz      =  float(clean[mapping['rel_height']].mean())
            ext      =  float(clean[mapping['extension']].mean())
            spin_rate = float(clean[mapping['spin_rate']].mean()) if 'spin_rate' in mapping else None
            spin_axis = float(clean[mapping['spin_axis']].mean()) if 'spin_axis' in mapping else None
            vaa = float(clean[mapping['vaa']].mean()) if 'vaa' in mapping else None

            # release_pos_x: prefer the trajectory-frame x0 (already Statcast-correct
            # and guaranteed consistent with vx0/ax). Only fall back to the flipped
            # RelSide if x0 isn't present.
            if 'x0' in mapping:
                rpx = float(clean[mapping['x0']].mean())
            else:
                rpx = -float(clean[mapping['rel_side']].mean())  # flip RelSide

            # Active spin (model's "spin_efficiency" field is really active spin in rpm)
            if 'true_spin' in mapping:
                active_spin = float(clean[mapping['true_spin']].mean())
            elif 'spin_eff' in mapping and spin_rate is not None:
                eff = float(clean[mapping['spin_eff']].mean())
                eff = eff / 100.0 if eff > 1.5 else eff  # handle pct vs ratio
                active_spin = spin_rate * eff
            elif spin_rate is not None:
                active_spin = spin_rate * 0.90  # assume 90% efficiency as fallback
            else:
                active_spin = None

            # Trajectory features — NOT flipped (already Statcast convention)
            fixed = {
                'release_pos_x': rpx,
                'release_pos_z': rpz,
                'release_extension': ext,
                'release_pos_y': 60.5 - ext,
                'vaa_mean': vaa,
                'spin_efficiency': active_spin,
            }
            if has_traj:
                fixed['vx0'] = float(clean[mapping['vx0']].mean())
                fixed['vy0'] = float(clean[mapping['vy0']].mean())
                fixed['vz0'] = float(clean[mapping['vz0']].mean())
                fixed['ax']  = float(clean[mapping['ax0']].mean())
                fixed['ay']  = float(clean[mapping['ay0']].mean())
                fixed['az']  = float(clean[mapping['az0']].mean())
            else:
                # Derive from shape later in app; store None so tunneling falls back
                for k in ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']:
                    fixed[k] = None

            pitches[pt] = {
                'fixed': fixed,
                'semi_fixed': {
                    'velo_mean': velo,
                    'velo_lo': round(velo - 1.5, 1),
                    'velo_hi': round(velo + 1.5, 1),
                    'spin_axis_mean': spin_axis,
                    'spin_rate_mean': spin_rate,
                },
                'optimizable': {
                    'pfx_x_mean': hb_mean,
                    'pfx_z_mean': ivb_mean,
                    'pfx_x_lo': None, 'pfx_x_hi': None,  # filled by comp lookup
                    'pfx_z_lo': None, 'pfx_z_hi': None,
                },
                'n_pitches': len(clean),
                'n': len(clean),
                'grades': {}, 'grades_rhh': {}, 'grades_lhh': {},
            }

        if pitches:
            profiles[synthetic_id] = {
                'player_name': name,
                'p_throws': p_throws,
                'pitches': pitches,
                'is_synthetic': True,
            }
            synthetic_id += 1

    if not profiles:
        messages.append("No pitchers with sufficient data found (need >=5 pitches of a type).")
    else:
        total_pitches = sum(len(p['pitches']) for p in profiles.values())
        messages.append(f"Built {len(profiles)} pitcher profile(s) with {total_pitches} pitch type(s) total.")

    return profiles, messages
