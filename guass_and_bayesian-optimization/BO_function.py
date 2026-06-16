import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
from scipy.stats.qmc import Sobol
import hashlib
import warnings
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import os

warnings.filterwarnings('ignore')
np.random.seed(114514)   # Can be commented out for different results


# ==================== 1. Acquisition Functions ====================
class AcquisitionFunction:
    @staticmethod
    def ucb(mean, std, kappa=2.576):
        return mean + kappa * std

    @staticmethod
    def ei(mean, std, y_best):
        if std == 0:
            return 0.0
        Z = (mean - y_best) / std
        return (mean - y_best) * norm.cdf(Z) + std * norm.pdf(Z)


# ==================== 2. Parameter Configuration Validation (Generate legal value list) ====================
class ParameterConfigValidator:
    @staticmethod
    def validate_config(param_config: Dict) -> Dict:
        validated = {}
        for param, cfg in param_config.items():
            lo, hi = cfg["bounds"]
            if lo >= hi:
                raise ValueError(f"Parameter {param}: lower bound {lo} must be less than upper bound {hi}")
            step = float(cfg["step"])
            if cfg.get("dtype") == "int":
                values = np.arange(lo, hi + step/2, step).astype(int)
            else:
                values = np.arange(lo, hi + step/2, step)
            values = values[(values >= lo) & (values <= hi)]
            validated[param] = {
                "bounds": tuple(cfg["bounds"]),
                "step": step,
                "dtype": cfg.get("dtype", "float"),
                "values": values.tolist()
            }
        return validated


# ==================== 3. Data Management ====================
class DataManager:
    def __init__(self, param_config: Dict):
        self.param_keys = list(param_config.keys())

    def load_scored_data(self, filepath: str, target_col: str) -> pd.DataFrame:
        df = pd.read_excel(filepath)
        if target_col not in df.columns:
            raise ValueError(f"File missing '{target_col}' column, please run preprocessing first")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])
        for k in self.param_keys:
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors='coerce')
        df = df.dropna(subset=self.param_keys)
        df['_hash'] = df.apply(lambda r: hashlib.md5(
            str([float(r[k]) for k in self.param_keys]).encode()).hexdigest(), axis=1)
        df = df.drop_duplicates(subset=['_hash']).reset_index(drop=True)
        return df

    def save_suggestions(self, suggestions, savepath, existing_df=None):
        df_new = pd.DataFrame(suggestions)
        start = 1
        if existing_df is not None and 'ExperimentID' in existing_df.columns:
            existing_df['ExperimentID'] = pd.to_numeric(existing_df['ExperimentID'], errors='coerce')
            if not existing_df['ExperimentID'].isna().all():
                start = int(existing_df['ExperimentID'].max()) + 1
        df_new.insert(0, 'ExperimentID', range(start, start + len(suggestions)))
        df_new['GenerationTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        df_new.to_excel(savepath, index=False)
        return df_new


# ==================== 4. Preprocessing: Compute optimization objective + peak soft constraint ====================
def preprocess_multi_objective(
        input_file: str,
        output_file: str,
        w_fwhm=0.7,
        w_peak=0,
        w_pl=0,
        fwhm_col='FWHM',
        peak_col='PeakPosition',
        pl_col='PLIntensity',
        peak_min=1.58,
        peak_max=1.60,
        penalty_strength=10.0   # Reduced default penalty strength
):
    df = pd.read_excel(input_file)

    # Continue sequence number if original column exists
    if 'Index' in df.columns and 'ExperimentID' not in df.columns:
        df['ExperimentID'] = df['Index']

    # Force numeric conversion
    for col in [fwhm_col, peak_col, pl_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[fwhm_col, peak_col, pl_col])

    # FWHM normalization (smaller is better -> larger score)
    fwhm_raw = df[fwhm_col].values.reshape(-1, 1)
    if fwhm_raw.min() == fwhm_raw.max():
        fwhm_score = np.ones(len(df))
    else:
        scaler_fwhm = MinMaxScaler()
        fwhm_norm = scaler_fwhm.fit_transform(fwhm_raw).ravel()
        fwhm_score = 1 - fwhm_norm

    peak_score = df[peak_col].values  # larger is better
    # PL intensity normalization (larger is better)
    pl_raw = df[pl_col].values.reshape(-1, 1)
    if pl_raw.min() == pl_raw.max():
        pl_score = np.ones(len(df))
    else:
        scaler_pl = MinMaxScaler()
        pl_score = scaler_pl.fit_transform(pl_raw).ravel()

    total = w_fwhm + w_peak + w_pl
    if total == 0:
        raise ValueError("Sum of weights cannot be zero")
    w_fwhm /= total
    w_peak /= total
    w_pl /= total
    print(f"Normalized weights: FWHM={w_fwhm:.3f}, Peak={w_peak:.3f}, PL={w_pl:.3f}")

    base_score = w_fwhm * fwhm_score + w_peak * peak_score + w_pl * pl_score

    # Peak soft constraint penalty (quadratic)
    peak_values = df[peak_col].values
    penalty = np.zeros_like(peak_values)
    below = peak_values < peak_min
    above = peak_values > peak_max
    penalty[below] = (peak_min - peak_values[below]) ** 2
    penalty[above] = (peak_values[above] - peak_max) ** 2
    df['Objective'] = base_score - penalty_strength * penalty

    df['Objective'] = pd.to_numeric(df['Objective'], errors='coerce')
    df = df.dropna(subset=['Objective'])

    df.to_excel(output_file, index=False)
    print(f"Peak soft constraint interval [{peak_min}, {peak_max}], penalty strength {penalty_strength}")
    print(f"Optimization target range: {df['Objective'].min():.4f} ~ {df['Objective'].max():.4f}")
    if np.any(penalty > 0):
        print(f"{np.sum(penalty>0)} samples penalized due to peak outside range")
    return output_file


# ==================== 5. Bayesian Optimization Main Class ====================
class BayesianOptimizer:
    def __init__(self, param_config: Dict):
        self.param_config = ParameterConfigValidator.validate_config(param_config)
        self.param_keys = list(self.param_config.keys())
        self.data_manager = DataManager(self.param_config)
        self.scaler = StandardScaler()

    def _compute_adaptive_bounds(self, df: pd.DataFrame, expand_ratio: float = 0.2) -> Dict[str, tuple]:
        """Compute adaptive bounds based on the given dataset (top-K historical samples) using legal value index expansion"""
        adaptive = {}
        for p in self.param_keys:
            values = df[p].values
            if len(values) == 0:
                adaptive[p] = self.param_config[p]["bounds"]
                continue
            legal_vals = np.array(self.param_config[p]["values"])
            lo_hist = np.min(values)
            hi_hist = np.max(values)
            # Find indices of historical data in legal value list
            idx_lo = np.argmin(np.abs(legal_vals - lo_hist))
            idx_hi = np.argmin(np.abs(legal_vals - hi_hist))
            lo_val = legal_vals[idx_lo]
            hi_val = legal_vals[idx_hi]
            rng = hi_val - lo_val
            if rng == 0:
                rng = 1.0
            n_vals = len(legal_vals)
            # Expand index range
            expand_steps = int(expand_ratio * (idx_hi - idx_lo + 1))
            new_idx_lo = max(0, idx_lo - expand_steps)
            new_idx_hi = min(n_vals-1, idx_hi + expand_steps)
            final_lo = legal_vals[new_idx_lo]
            final_hi = legal_vals[new_idx_hi]
            adaptive[p] = (final_lo, final_hi)
            print(f"Parameter '{p}': adaptive bounds [{final_lo:.2f}, {final_hi:.2f}] (based on range [{lo_val:.2f}, {hi_val:.2f}], expansion {expand_ratio*100:.0f}%)")
        return adaptive

    def _generate_candidates_sobol(self, n_candidates: int, bounds_dict: Dict[str, tuple]) -> List[Dict]:
        """Generate global exploration candidate points using Sobol sequence, strictly discretized by step size"""
        dim = len(self.param_keys)
        lo = np.array([bounds_dict[p][0] for p in self.param_keys])
        hi = np.array([bounds_dict[p][1] for p in self.param_keys])
        steps = np.array([self.param_config[p]["step"] for p in self.param_keys])

        sobol = Sobol(d=dim, scramble=True, seed=np.random.randint(0, 1e6))
        samples_raw = sobol.random(n_candidates)
        samples_scaled = lo + samples_raw * (hi - lo)

        candidates = []
        for i in range(n_candidates):
            point = {}
            for j, p in enumerate(self.param_keys):
                val = samples_scaled[i, j]
                step = steps[j]
                lo_j = lo[j]
                # Robust discretization: compute closest integer step count
                n_steps = int(round((val - lo_j) / step))
                al = lo_j + n_steps * step
                al = np.clip(al, lo_j, hi[j])
                if self.param_config[p]["dtype"] == "int":
                    al = int(al)
                point[p] = al
            candidates.append(point)
        # Deduplicate
        unique = []
        seen = set()
        for c in candidates:
            key = tuple(c.items())
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _generate_candidates_from_topk(self, df: pd.DataFrame, top_k: int, n_candidates: int,
                                       bounds_dict: Dict[str, tuple], iteration: int = 0, max_iter: int = 10) -> List[Dict]:
        """
        Generate candidate points from the neighborhood of top-K historical samples (exploitation)
        Supports dynamic step size: large (±3) in early iterations, ±2 in middle, ±1 in late
        """
        if top_k <= 0 or len(df) == 0:
            return []
        top_df = df.nlargest(top_k, 'Objective')
        # Dynamic max step based on iteration progress
        progress = min(iteration / max(max_iter, 1), 1.0)
        if progress < 0.3:
            max_step = 3
        elif progress < 0.7:
            max_step = 2
        else:
            max_step = 1
        # Can also adjust based on GP uncertainty (optional)
        candidates = []
        for _ in range(n_candidates):
            base = top_df.sample(1).iloc[0]
            point = {}
            for p in self.param_keys:
                legal_vals = np.array(self.param_config[p]["values"])
                lo_b, hi_b = bounds_dict[p]
                allowed_vals = legal_vals[(legal_vals >= lo_b) & (legal_vals <= hi_b)]
                if len(allowed_vals) == 0:
                    allowed_vals = legal_vals  # fallback
                base_val = base[p]
                idx = np.argmin(np.abs(allowed_vals - base_val))
                move = np.random.choice(range(-max_step, max_step+1))
                new_idx = idx + move
                new_idx = np.clip(new_idx, 0, len(allowed_vals)-1)
                new_val = allowed_vals[new_idx]
                if self.param_config[p]["dtype"] == "int":
                    new_val = int(new_val)
                point[p] = new_val
            candidates.append(point)
        # Deduplicate
        unique = []
        seen = set()
        for c in candidates:
            key = tuple(c.items())
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _diversity_filter(self, suggestions: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Remove overly similar recommendation points (based on proportion of identical parameters)"""
        if len(suggestions) <= 1:
            return suggestions
        filtered = []
        for s in suggestions:
            similar = False
            for f in filtered:
                same = sum(1 for p in s if s[p] == f[p]) / len(s)
                if same > similarity_threshold:
                    similar = True
                    break
            if not similar:
                filtered.append(s)
        # If filtered result is too small, replenish from original list (preserve order)
        if len(filtered) < len(suggestions):
            for s in suggestions:
                if s not in filtered:
                    filtered.append(s)
                    if len(filtered) == len(suggestions):
                        break
        return filtered

    def optimize(
        self,
        filepath: str,
        savepath: str,
        target_col: str = "Objective",
        acq_type: str = "ei",
        kappa: float = 2.576,
        n_points: int = 10,
        n_candidates: int = 50000,
        use_adaptive_bounds: bool = True,
        adaptive_bound_expand: float = 0.2,
        top_k_exploit: int = 10,
        exploit_ratio: float = 0.5,
        iteration: int = 0,
        max_iter: int = 10,
        diversity_threshold: float = 0.7,
    ) -> pd.DataFrame:
        # Load data
        df = self.data_manager.load_scored_data(filepath, target_col)
        if len(df) == 0:
            raise ValueError("No valid samples after preprocessing!")

        # Automatically switch acquisition function in later stages
        if len(df) > 30 and df[target_col].std() < 0.01:
            acq_type = "ucb"
            kappa = 4.0
            print("Detected late optimization (many samples and flat objective), auto-switch to UCB acquisition, kappa=4.0")

        # Train GP
        gp = self._train_gp(df, target_col)

        # Determine search bounds
        if use_adaptive_bounds and len(df) >= 3:
            if top_k_exploit > 0:
                subset_df = df.nlargest(top_k_exploit, 'Objective')
                if len(subset_df) < 2:
                    subset_df = df
                    print(f"Note: fewer than 2 samples in top-{top_k_exploit} historical best, using all data for adaptive bounds")
            else:
                subset_df = df
            bounds_dict = self._compute_adaptive_bounds(subset_df, expand_ratio=adaptive_bound_expand)
        else:
            bounds_dict = {p: self.param_config[p]["bounds"] for p in self.param_keys}
            print("Adaptive bounds not enabled, using original parameter ranges")

        # Hybrid candidate generation: force at least 20% global exploration
        min_explore_ratio = 0.2
        n_exploit = int(n_candidates * exploit_ratio) if top_k_exploit > 0 else 0
        n_explore = n_candidates - n_exploit
        if n_explore / n_candidates < min_explore_ratio:
            n_explore = int(n_candidates * min_explore_ratio)
            n_exploit = n_candidates - n_explore
            print(f"Forced global exploration ratio to {min_explore_ratio*100:.0f}% (explore {n_explore}, exploit {n_exploit})")

        # Generate candidates
        candidates_explore = self._generate_candidates_sobol(n_explore, bounds_dict) if n_explore > 0 else []
        candidates_exploit = self._generate_candidates_from_topk(df, top_k_exploit, n_exploit, bounds_dict, iteration, max_iter) if n_exploit > 0 else []
        candidates = candidates_explore + candidates_exploit
        print(f"Generated candidates: global exploration {len(candidates_explore)}, local exploitation {len(candidates_exploit)} (total {len(candidates)})")

        # Exclude points already in history
        existing_set = set(tuple(row[k] for k in self.param_keys) for _, row in df.iterrows())
        candidates = [c for c in candidates if tuple(c.values()) not in existing_set]

        if len(candidates) == 0:
            print("Warning: all candidate points already exist in historical data, will regenerate with larger neighborhood steps")
            # Regenerate exploitation points with larger steps
            candidates_exploit = self._generate_candidates_from_topk(df, top_k_exploit, n_candidates, bounds_dict, iteration, max_iter)
            candidates = candidates_exploit
            candidates = [c for c in candidates if tuple(c.values()) not in existing_set]
            if len(candidates) == 0:
                raise RuntimeError("Unable to generate new candidate points, please check parameter space or historical data")

        # Compute acquisition values
        acq_values = self._compute_acquisition(gp, candidates, acq_type, kappa, df[target_col].max())
        sorted_idx = np.argsort(acq_values)[::-1][:n_points*2]  # take extra for diversity filtering
        top_candidates = [candidates[i] for i in sorted_idx]

        # Diversity filtering
        suggestions = self._diversity_filter(top_candidates, similarity_threshold=diversity_threshold)
        if len(suggestions) < n_points:
            # Fill missing points (in acquisition order)
            for c in top_candidates:
                if c not in suggestions:
                    suggestions.append(c)
                    if len(suggestions) == n_points:
                        break
        suggestions = suggestions[:n_points]

        # Save results
        df_suggest = self.data_manager.save_suggestions(suggestions, savepath, df)
        df_suggest = self._add_predictions(gp, df_suggest)

        print(f"\n===== Generated {len(suggestions)} recommendations =====")
        cols = ['ExperimentID'] + self.param_keys + ['PredictedMean', 'PredictedStd', 'PredictedRank', 'GenerationTime']
        df_suggest = df_suggest[[c for c in cols if c in df_suggest.columns]]
        print(df_suggest.to_string(index=False))
        return df_suggest

    def _train_gp(self, df: pd.DataFrame, target_col: str):
        X = df[self.param_keys].values
        y = df[target_col].values
        if len(X) < 2:
            raise ValueError(f"Too few valid samples ({len(X)}), cannot train GP.")
        X_scaled = self.scaler.fit_transform(X)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) \
                 * Matern(length_scale=np.ones(X.shape[1]),
                          length_scale_bounds=(1e-2, 1e3),
                          nu=2.5) \
                 + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X_scaled, y)
        return gp

    def _compute_acquisition(self, gp, candidates, acq_type, kappa, y_best):
        X_cand = np.array([[c[p] for p in self.param_keys] for c in candidates])
        X_scaled = self.scaler.transform(X_cand)
        mean, std = gp.predict(X_scaled, return_std=True)
        std = np.clip(std, 1e-9, None)
        if acq_type.lower() == 'ucb':
            return mean + kappa * std
        elif acq_type.lower() == 'ei':
            return np.array([AcquisitionFunction.ei(m, s, y_best) for m, s in zip(mean, std)])
        else:
            raise ValueError(f"Unknown acquisition function type: {acq_type}")

    def _add_predictions(self, gp, df_suggest):
        X = df_suggest[self.param_keys].values
        X_scaled = self.scaler.transform(X)
        mean, std = gp.predict(X_scaled, return_std=True)
        df_suggest['PredictedMean'] = mean
        df_suggest['PredictedStd'] = std
        df_suggest['PredictedRank'] = pd.Series(mean).rank(ascending=False).astype(int).values
        print("\n=== Prediction performance evaluation ===")
        print(df_suggest[['ExperimentID', 'PredictedMean', 'PredictedStd', 'PredictedRank']]
              .sort_values('PredictedMean', ascending=False).to_string(index=False))
        return df_suggest


# ==================== 6. Main Program ====================
if __name__ == "__main__":
    # -------------------- File paths --------------------
    RAW_DATA = "training_data/version1/initial_data_with_features.xlsx"
    PROCESSED_DATA = "training_data/version1/initial_data_with_features_scored.xlsx"
    SAVE_RESULT = "newProcesses/version1/Round1.xlsx"

    # -------------------- Parameter configuration (strict step sizes) --------------------
    param_config = {
        "MACl":             {"bounds": (5, 30),   "step": 5,   "dtype": "int"},
        "PrecursorVolume":  {"bounds": (20, 60),  "step": 5,   "dtype": "int"},
        "AntiSolventVolume":{"bounds": (100, 600),"step": 50,  "dtype": "int"},
        "AntiSolventTime":  {"bounds": (5, 15),   "step": 1,   "dtype": "int"},
        "SpinTime2":        {"bounds": (20, 50),  "step": 5,   "dtype": "int"},
        "SpinSpeed1":       {"bounds": (500, 2000),"step": 500, "dtype": "int"},
        "SpinSpeed2":       {"bounds": (2000, 6000),"step": 500,"dtype": "int"},
        "Acceleration":     {"bounds": (1000, 5000),"step": 500, "dtype": "int"},
        "AnnealingTime":    {"bounds": (1, 20),   "step": 1,   "dtype": "int"},
    }

    # -------------------- Multi-objective weights --------------------
    W_FWHM = 0.9
    W_PEAK = 0.1
    W_PL   = 0.0

    # -------------------- Bayesian optimization parameters --------------------
    ACQ_TYPE = "ei"               # Use ei initially, auto-switch later
    KAPPA = 3.0                   # UCB exploration coefficient (only effective for UCB)
    N_POINTS = 10
    N_CANDIDATES = 80000

    TOP_K_EXPLOIT = 5            # Use top 10 historical experiments for exploitation
    EXPLOIT_RATIO = 0.6           # 40% exploitation, 60% exploration (at least 20% exploration forced)
    USE_ADAPTIVE_BOUNDS = True
    ADAPTIVE_BOUND_EXPAND = 0.1   # Expand bounds outward by 50% (better exploration)

    # Dynamic step parameters (iteration round; not necessary to pass externally, default max_iter=10 controls step decay)
    ITERATION = 3                 # Current optimization round (starting from 0), can be passed externally or read from filename
    MAX_ITER = 10                 # Expected total rounds (used to control step decay)

    # Diversity filtering threshold
    DIVERSITY_THRESHOLD = 0.6     # 70% identical parameters considered duplicate

    # -------------------- Peak soft constraint --------------------
    PEAK_MIN = 1.50
    PEAK_MAX = 1.61
    PENALTY_STRENGTH = 100.0       # Moderately reduce penalty strength to avoid over-domination

    # -------------------- Column names (now in English) --------------------
    FWHM_COL = 'FWHM'
    PEAK_COL = 'PeakPosition'
    PL_COL = 'PLIntensity'
    TARGET_COL = 'Objective'

    # -------------------- Run preprocessing --------------------
    preprocess_multi_objective(
        input_file=RAW_DATA,
        output_file=PROCESSED_DATA,
        w_fwhm=W_FWHM,
        w_peak=W_PEAK,
        w_pl=W_PL,
        fwhm_col=FWHM_COL,
        peak_col=PEAK_COL,
        pl_col=PL_COL,
        peak_min=PEAK_MIN,
        peak_max=PEAK_MAX,
        penalty_strength=PENALTY_STRENGTH
    )

    # -------------------- Run optimization --------------------
    optimizer = BayesianOptimizer(param_config)
    result = optimizer.optimize(
        filepath=PROCESSED_DATA,
        savepath=SAVE_RESULT,
        target_col=TARGET_COL,
        acq_type=ACQ_TYPE,
        kappa=KAPPA,
        n_points=N_POINTS,
        n_candidates=N_CANDIDATES,
        use_adaptive_bounds=USE_ADAPTIVE_BOUNDS,
        adaptive_bound_expand=ADAPTIVE_BOUND_EXPAND,
        top_k_exploit=TOP_K_EXPLOIT,
        exploit_ratio=EXPLOIT_RATIO,
        iteration=ITERATION,
        max_iter=MAX_ITER,
        diversity_threshold=DIVERSITY_THRESHOLD,
    )

    print(f"\nOptimization complete, generated {len(result)} recommendation points, saved to {SAVE_RESULT}")
