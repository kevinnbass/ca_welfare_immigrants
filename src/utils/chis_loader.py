"""
CHIS (California Health Interview Survey) data loading utilities.

Handles loading CHIS Public Use Files and computing weighted estimates.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CHISLoader:
    """Load and process CHIS Adult Public Use Files."""

    # CHIS uses 80 replicate weights with JK1 method
    N_REPLICATES = 80
    MAIN_WEIGHT = "RAKEDW0"
    REPLICATE_PREFIX = "RAKEDW"

    # Variable mappings
    CITIZEN_VAR = "CITIZEN2"
    CITIZEN_LABELS = {
        1.0: "US_BORN",
        2.0: "NATURALIZED",
        3.0: "NONCITIZEN",
    }

    PROGRAM_VARS = {
        "medicaid": {"var": "INSMC", "condition": lambda x: x == 1},
        "food_stamps": {"var": "AL5V2", "condition": lambda x: x == 1},
    }

    def __init__(self, data_dir: Path):
        """
        Initialize CHIS loader.

        Args:
            data_dir: Directory containing CHIS data files
        """
        self.data_dir = Path(data_dir)
        self._df: Optional[pd.DataFrame] = None
        self._meta = None

    def load(self, year: int = 2023) -> pd.DataFrame:
        """
        Load CHIS adult data for specified year.

        Args:
            year: CHIS year

        Returns:
            DataFrame with CHIS data
        """
        import pyreadstat

        file_path = self.data_dir / f"chis_adult_{year}.sas7bdat"

        if not file_path.exists():
            raise FileNotFoundError(f"CHIS data file not found: {file_path}")

        logger.info(f"Loading CHIS {year} data from {file_path}")
        self._df, self._meta = pyreadstat.read_sas7bdat(file_path)

        logger.info(f"Loaded {len(self._df):,} records with {len(self._df.columns)} variables")

        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Get loaded DataFrame."""
        if self._df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._df

    def get_replicate_weight_columns(self) -> list[str]:
        """Get list of replicate weight column names."""
        return [f"{self.REPLICATE_PREFIX}{i}" for i in range(1, self.N_REPLICATES + 1)]

    def compute_rate_with_se(
        self,
        program: str,
        citizenship_group: Optional[float] = None,
    ) -> dict:
        """
        Compute weighted rate with standard error using replicate weights.

        Uses JK1 (Jackknife 1) variance formula:
        Var = sum_r (theta_r - theta)^2

        Args:
            program: Program key (medicaid, food_stamps)
            citizenship_group: CITIZEN2 value (1, 2, or 3) or None for all

        Returns:
            Dict with estimate, se, ci_lower, ci_upper, n_unweighted, n_weighted
        """
        df = self.df.copy()

        # Apply citizenship filter
        if citizenship_group is not None:
            df = df[df[self.CITIZEN_VAR] == citizenship_group]

        prog_info = self.PROGRAM_VARS[program]
        var_name = prog_info["var"]
        condition = prog_info["condition"]

        # Filter to valid responses (not -1 which is "not asked")
        valid_mask = df[var_name] > 0
        df_valid = df[valid_mask]

        if len(df_valid) == 0:
            return {
                "estimate": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_unweighted": 0,
                "n_weighted": 0,
            }

        # Create binary indicator
        indicator = condition(df_valid[var_name]).astype(int)
        weights = df_valid[self.MAIN_WEIGHT]

        # Main estimate
        theta = np.average(indicator, weights=weights)

        # Replicate estimates for JK1 variance
        rep_cols = self.get_replicate_weight_columns()
        theta_reps = []

        for rep_col in rep_cols:
            if rep_col in df_valid.columns:
                theta_r = np.average(indicator, weights=df_valid[rep_col])
                theta_reps.append(theta_r)

        theta_reps = np.array(theta_reps)

        # JK1 variance: Var = sum (theta_r - theta)^2
        variance = np.sum((theta_reps - theta) ** 2)
        se = np.sqrt(variance)

        # 95% CI
        z = 1.96
        ci_lower = theta - z * se
        ci_upper = theta + z * se

        return {
            "estimate": theta,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_unweighted": len(df_valid),
            "n_weighted": weights.sum(),
        }

    def compute_all_rates(self) -> pd.DataFrame:
        """
        Compute rates for all programs and citizenship groups.

        Returns:
            DataFrame with rates by program and group
        """
        results = []

        for program in self.PROGRAM_VARS.keys():
            for cit_code, cit_label in self.CITIZEN_LABELS.items():
                rate_info = self.compute_rate_with_se(program, cit_code)
                results.append({
                    "source": "CHIS",
                    "year": 2023,
                    "program": program,
                    "group": cit_label,
                    **rate_info,
                })

            # Also compute overall rate
            rate_info = self.compute_rate_with_se(program, None)
            results.append({
                "source": "CHIS",
                "year": 2023,
                "program": program,
                "group": "ALL",
                **rate_info,
            })

        return pd.DataFrame(results)
