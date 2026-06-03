"""
stuff_model_interface.py
========================
Pluggable interface for stuff model predictions.

To use a custom stuff model with the Shape Optimizer:
  1. Subclass StuffModelInterface
  2. Implement predict_rv(features: dict) -> float
  3. Implement normalize(pred_rv, pitch_type, p_throws, stand) -> dict
  4. Pass your implementation to ShapeOptimizer()

The Shape Optimizer calls this interface exclusively —
it never touches model internals directly.
"""

import numpy as np
from abc import ABC, abstractmethod


class StuffModelInterface(ABC):
    """
    Abstract base class for stuff model predictions.
    Any stuff model can be plugged into the Shape Optimizer
    by implementing these two methods.
    """

    @abstractmethod
    def predict_rv(self, features: dict) -> float:
        """
        Predict run value for a single pitch given its characteristics.

        Parameters
        ----------
        features : dict with keys:
            release_speed       : float  — velocity (mph)
            pfx_x_in            : float  — horizontal break (inches, catcher's POV)
            pfx_z_in            : float  — vertical break (inches)
            release_pos_x       : float  — arm slot horizontal (ft, catcher's POV)
            release_pos_z       : float  — arm slot vertical (ft)
            release_extension   : float  — extension (ft)
            velo_diff_fb        : float  — velo delta vs primary fastball
            pfx_x_diff_fb       : float  — h-break delta vs primary fastball
            pfx_z_diff_fb       : float  — v-break delta vs primary fastball
            vaa                 : float  — vertical approach angle (degrees, FF/SI only)
            spin_axis           : float  — spin axis (0-360 degrees)
            spin_efficiency_raw : float  — active spin rate (rpm)
            ssw_interaction     : float  — spin_axis_rad * spin_efficiency (SSW proxy)

        Returns
        -------
        float : predicted run value per pitch
                negative = good for pitcher (matches Fangraphs convention)
        """
        pass

    @abstractmethod
    def normalize(self, pred_rv: float, pitch_type: str,
                  p_throws: str, stand: str) -> dict:
        """
        Convert raw predicted RV to normalized grades.

        Parameters
        ----------
        pred_rv    : float — raw model prediction from predict_rv()
        pitch_type : str   — e.g. 'FF', 'SL', 'CH'
        p_throws   : str   — 'R' or 'L'
        stand      : str   — 'R' or 'L' (batter handedness)

        Returns
        -------
        dict with keys:
            stuff_plus   : float — within-pitch-type grade (100 = league avg)
            arsenal_plus : float — cross-arsenal grade (100 = league avg)
            contact_plus : float — contact suppression grade (100 = league avg, optional)
        """
        pass


class PitchScoutStuffModel(StuffModelInterface):
    """
    Default Pitch Scout stuff model implementation.
    Wraps the XGBoost models trained in the notebook.
    """

    STUFF_SCALE = 0.15  # RV units per 100 Stuff+ points

    def __init__(self, stuff_models: dict, norm_tables: dict,
                 p_throws: str, stand: str, pitch_type: str):
        """
        Parameters
        ----------
        stuff_models : dict — keyed (p_throws, stand, pitch_type)
        norm_tables  : dict — normalization tables from notebook
        p_throws     : str  — pitcher handedness for this instance
        stand        : str  — batter handedness for this instance
        pitch_type   : str  — pitch type for this instance
        """
        self.p_throws    = p_throws
        self.stand       = stand
        self.pitch_type  = pitch_type
        self.norm_tables = norm_tables

        key = (p_throws, stand, pitch_type)
        self._mpt = stuff_models.get(key, {})

    def _predict(self, model, X):
        """Handle both Booster and XGBRegressor predict APIs."""
        if hasattr(model, 'inplace_predict'):
            return model.inplace_predict(X)
        return model.predict(X)

    def predict_rv(self, features: dict) -> float:
        if 'rv' not in self._mpt:
            return 0.0
        feat_cols = self._mpt['feature_cols']
        X = np.array(
            [features.get(c, 0.0) or 0.0 for c in feat_cols],
            dtype=float
        ).reshape(1, -1)
        return float(self._predict(self._mpt['rv'], X)[0])

    def normalize(self, pred_rv: float, pitch_type: str,
                  p_throws: str, stand: str) -> dict:
        norm_key = (p_throws, stand, pitch_type)
        lg_mean  = self.norm_tables['rv_mean'].get(norm_key)
        lg_all   = self.norm_tables['rv_mean_all_by_pt'].get(pitch_type)

        result = {}
        if lg_mean is not None:
            result['stuff_plus'] = round(
                100 + 100 * (lg_mean - pred_rv) / self.STUFF_SCALE, 1)
        if lg_all is not None:
            result['arsenal_plus'] = round(
                100 + 100 * (lg_all - pred_rv) / self.STUFF_SCALE, 1)

        # Contact+ (optional — uses hard_hit and xwoba models)
        if 'hard_hit' in self._mpt and 'xwoba' in self._mpt:
            feat_cols = self._mpt['feature_cols']
            # We'd need X here — Contact+ is computed separately in the optimizer
            pass

        return result

    def predict_grades_full(self, features: dict) -> dict:
        """
        Convenience method: predict RV, normalize, and compute Contact+.
        Returns full grades dict.
        """
        if not self._mpt:
            return {}

        feat_cols = self._mpt['feature_cols']
        X = np.array(
            [features.get(c, 0.0) or 0.0 for c in feat_cols],
            dtype=float
        ).reshape(1, -1)

        result = {}
        norm_key = (self.p_throws, self.stand, self.pitch_type)

        if 'rv' in self._mpt:
            pred_rv    = float(self._predict(self._mpt['rv'], X)[0])
            lg_mean    = self.norm_tables['rv_mean'].get(norm_key, 0.0)
            lg_all     = self.norm_tables['rv_mean_all_by_pt'].get(self.pitch_type, lg_mean)
            correction = self.norm_tables.get('stuff_plus_correction', 0.0)
            result['pred_rv']      = pred_rv
            result['stuff_plus']   = round(100 + 100 * (lg_mean - pred_rv) / self.STUFF_SCALE - correction, 1)
            result['arsenal_plus'] = round(100 + 100 * (lg_all  - pred_rv) / self.STUFF_SCALE - correction, 1)

        if 'hard_hit' in self._mpt and 'xwoba' in self._mpt:
            pred_hh = float(np.clip(self._predict(self._mpt['hard_hit'], X)[0], 0, 1))
            pred_xw = float(np.clip(self._predict(self._mpt['xwoba'],    X)[0], 0, 1))
            lg_hh   = self.norm_tables['hh_mean'].get(norm_key, pred_hh)
            lg_xw   = self.norm_tables['xwoba_mean'].get(norm_key, pred_xw)
            hh_plus = (100 * lg_hh / pred_hh) if pred_hh > 0 else 100.0
            xw_plus = (100 * lg_xw / pred_xw) if pred_xw > 0 else 100.0
            result['pred_hard_hit'] = round(pred_hh, 4)
            result['pred_xwoba']    = round(pred_xw, 4)
            result['contact_plus']  = round((hh_plus + xw_plus) / 2, 1)

        return result
