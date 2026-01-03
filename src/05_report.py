"""
Generate final report with visualizations.

This script:
1. Loads computed rate estimates
2. Creates visualization figures
3. Generates markdown report

Usage:
    python -m src.05_report [--year YEAR]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def load_results(year: int, suffix: str = "") -> pd.DataFrame:
    """Load rate estimation results."""
    filename = f"ca_rates_by_group_program_{year}{suffix}.csv"
    file_path = config.TABLES_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    return pd.read_csv(file_path)


def create_bar_chart(
    df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> Path:
    """
    Create bar chart of participation rates by group and program.

    Args:
        df: Results DataFrame
        year: ACS year
        output_dir: Directory for output

    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to main programs
    main_programs = ["medicaid", "snap", "ssi", "public_assistance"]
    df_plot = df[df["program"].isin(main_programs)].copy()

    # Pivot for plotting
    pivot = df_plot.pivot(index="program", columns="group", values="estimate")

    # Reorder
    group_order = ["US_BORN", "LEGAL_IMMIGRANT", "ILLEGAL"]
    group_order = [g for g in group_order if g in pivot.columns]
    program_order = ["medicaid", "snap", "ssi", "public_assistance"]
    program_order = [p for p in program_order if p in pivot.index]

    pivot = pivot.loc[program_order, group_order]

    # Create bar positions
    x = np.arange(len(program_order))
    width = 0.25

    colors = {"US_BORN": "#1f77b4", "LEGAL_IMMIGRANT": "#ff7f0e", "ILLEGAL": "#2ca02c"}
    labels = {"US_BORN": "US-Born", "LEGAL_IMMIGRANT": "Legal Immigrants", "ILLEGAL": "Illegal"}

    for i, group in enumerate(group_order):
        values = pivot[group].values * 100  # Convert to percentage
        bars = ax.bar(x + i * width, values, width, label=labels.get(group, group), color=colors.get(group))

        # Add error bars if available
        errors = df_plot[df_plot["group"] == group].set_index("program").loc[program_order, "se"].values * 100
        if not np.all(np.isnan(errors)):
            ax.errorbar(
                x + i * width,
                values,
                yerr=1.96 * errors,
                fmt="none",
                color="black",
                capsize=3,
            )

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Participation Rate (%)")
    ax.set_title(f"Welfare Program Participation by Immigration Status\nCalifornia {year}")

    program_labels = {
        "medicaid": "Medicaid/\nMedi-Cal",
        "snap": "SNAP/\nCalFresh",
        "ssi": "SSI",
        "public_assistance": "Public\nAssistance",
    }
    ax.set_xticks(x + width)
    ax.set_xticklabels([program_labels.get(p, p) for p in program_order])

    ax.legend(title="Immigration Status", loc="upper right")
    ax.set_ylim(0, max(pivot.max().max() * 100 * 1.3, 50))

    plt.tight_layout()

    output_path = output_dir / f"rates_by_group_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved bar chart: {output_path}")
    return output_path


def create_comparison_chart(
    observable_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> Path:
    """
    Create comparison chart showing observable vs imputed estimates.

    Args:
        observable_df: Observable status results
        imputed_df: Imputed status results
        year: ACS year
        output_dir: Directory for output

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    main_programs = ["medicaid", "snap", "ssi", "public_assistance"]
    program_labels = {
        "medicaid": "Medicaid/Medi-Cal",
        "snap": "SNAP/CalFresh",
        "ssi": "SSI",
        "public_assistance": "Public Assistance",
    }

    for idx, program in enumerate(main_programs):
        ax = axes[idx // 2, idx % 2]

        # Observable results
        obs = observable_df[observable_df["program"] == program]
        imp = imputed_df[imputed_df["program"] == program]

        x_obs = np.arange(len(obs))
        x_imp = np.arange(len(imp)) + 0.35

        obs_vals = obs["estimate"].values * 100
        imp_vals = imp["estimate"].values * 100

        ax.bar(x_obs, obs_vals, 0.35, label="Observable Status", alpha=0.7)
        ax.bar(x_imp, imp_vals, 0.35, label="Imputed Status", alpha=0.7)

        ax.set_ylabel("Rate (%)")
        ax.set_title(program_labels.get(program, program))
        ax.set_xticks(np.arange(max(len(obs), len(imp))) + 0.175)
        ax.set_xticklabels(obs["group"].values, rotation=45, ha="right")
        ax.legend(fontsize=8)

    plt.suptitle(f"Observable vs Imputed Status Estimates\nCalifornia {year}", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"observable_vs_imputed_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison chart: {output_path}")
    return output_path


def generate_report_markdown(
    df: pd.DataFrame,
    year: int,
    figure_paths: list[Path],
    admin_totals: Optional[pd.DataFrame] = None,
    bootstrap_results: Optional[dict] = None,
    is_pooled: bool = False,
) -> str:
    """
    Generate markdown report content.

    Args:
        df: Results DataFrame
        year: ACS year
        figure_paths: Paths to generated figures
        admin_totals: Optional administrative data totals for validation
        bootstrap_results: Optional bootstrap results with variance decomposition
        is_pooled: Whether this is a pooled multi-year analysis

    Returns:
        Markdown string
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Build tables
    def format_row(row):
        est = row["estimate"] * 100
        ci_low = row.get("ci_lower", np.nan) * 100
        ci_high = row.get("ci_upper", np.nan) * 100
        n = row.get("n_unweighted", 0)

        if row.get("suppressed", False):
            return "| [suppressed] | - | - |"

        ci_str = f"({ci_low:.1f}% - {ci_high:.1f}%)" if pd.notna(ci_low) else ""
        reliable = "" if row.get("reliable", True) else " *"

        return f"| {est:.1f}%{reliable} | {ci_str} | {int(n):,} |"

    # Create results table
    results_table = []
    programs = df["program"].unique()

    for program in programs:
        prog_df = df[df["program"] == program]
        label = prog_df.iloc[0].get("program_label", program)

        results_table.append(f"\n### {label}\n")
        results_table.append("| Group | Rate | 95% CI | n |")
        results_table.append("|-------|------|--------|---|")

        for _, row in prog_df.iterrows():
            group = row["group"]
            results_table.append(f"| {group} {format_row(row)}")

    results_table_str = "\n".join(results_table)

    # Generate validation sections
    admin_comparison_section = generate_admin_comparison_section(df, admin_totals, year)
    variance_decomposition_section = generate_variance_decomposition_section(df, bootstrap_results)
    limitations_section = generate_enhanced_limitations_section(
        has_admin_data=admin_totals is not None and not admin_totals.empty,
        has_bootstrap=bootstrap_results is not None,
        has_pooled=is_pooled,
    )

    # Figure references
    fig_refs = []
    for path in figure_paths:
        fig_refs.append(f"![{path.stem}](../outputs/figures/{path.name})")
    fig_refs_str = "\n\n".join(fig_refs)

    report = f"""# California Welfare Participation by Immigration Status

**Analysis Year:** {year}
**Report Generated:** {today}

## Executive Summary

This report presents estimates of welfare program participation rates in California
by immigration status. Using publicly available data from the American Community
Survey (ACS) and Survey of Income and Program Participation (SIPP), we estimate
participation rates for:

- **US-Born:** Persons born in the United States
- **Legal Immigrants:** Naturalized citizens and lawfully present noncitizens
- **Undocumented Immigrants:** Foreign-born noncitizens without legal status (imputed)

**Key findings:**

- [Summary findings would be inserted here based on results]

## Definitions

### Immigration Status Groups

| Group | Definition |
|-------|------------|
| US_BORN | Born in the United States (citizen at birth) |
| LEGAL_IMMIGRANT | Foreign-born, legally present (naturalized citizens + LPRs + visa holders) |
| ILLEGAL | Foreign-born noncitizen without legal status (imputed) |

### Welfare Programs

| Program | Description | Reference Period |
|---------|-------------|------------------|
| Medicaid/Medi-Cal | Public health insurance coverage | Current |
| SNAP/CalFresh | Food assistance (Supplemental Nutrition Assistance Program) | Past 12 months |
| SSI | Supplemental Security Income | Past 12 months |
| Public Assistance | Cash assistance income (TANF/CalWORKs proxy) | Past 12 months |

## Data Sources

1. **American Community Survey (ACS) PUMS** - U.S. Census Bureau
   - California {year} 1-Year Public Use Microdata Sample
   - Source: https://www.census.gov/programs-surveys/acs/microdata.html

2. **Survey of Income and Program Participation (SIPP)** - U.S. Census Bureau
   - Used to train legal status imputation model
   - Source: https://www.census.gov/sipp

3. **Pew Research Center** - Unauthorized immigrant population estimates
   - Used for calibration of imputed undocumented totals
   - Source: https://www.pewresearch.org/

## Methodology

### Status Imputation

Since the ACS does not directly identify undocumented status, we use a statistical
imputation approach:

1. **Model Training:** Using SIPP data (which contains partial legal status information),
   we train a classification model to predict P(undocumented | covariates).

2. **Prediction:** The model is applied to ACS noncitizens to obtain individual
   probabilities of being undocumented.

3. **Multiple Imputation:** We create {config.N_IMPUTATIONS} imputed datasets by drawing
   status from Bernoulli(p) for each noncitizen.

4. **Calibration:** Imputed totals are calibrated to match Pew Research state-level
   estimates of the unauthorized population.

5. **Combining Results:** Estimates are combined across imputations using Rubin's rules,
   which properly accounts for both sampling variance and imputation uncertainty.

### Uncertainty Quantification

Two sources of uncertainty are combined:

1. **Sampling Variance:** Estimated using ACS successive difference replication (SDR)
   with 80 replicate weights.

2. **Imputation Variance:** Between-imputation variance captured through Rubin's rules.

## Results

{results_table_str}

*Note: Estimates marked with * may be unreliable due to high coefficient of variation.*

## Visualizations

{fig_refs_str}

## Sensitivity Analysis

### Comparison of Observable vs. Imputed Status

Observable status (US-born vs. naturalized vs. noncitizen) is directly available in
the ACS. The imputed approach further distinguishes legal noncitizens from undocumented
noncitizens. Results by observable status are provided as a benchmark.

{admin_comparison_section}

{variance_decomposition_section}

{limitations_section}

## Reproducibility

This analysis was produced using the `ca_welfare_immigrants` pipeline.
See the repository README for instructions on reproducing these results.

### Software Versions

- Python: 3.11+
- Key packages: pandas, scikit-learn, statsmodels
- R: 4.0+ (survey package for variance estimation)

## References

1. U.S. Census Bureau. American Community Survey Public Use Microdata Sample.

2. U.S. Census Bureau. Survey of Income and Program Participation.

3. Pew Research Center. Unauthorized Immigrant Population Estimates.

4. Migration Policy Institute. Methodology for Assigning Legal Status to Noncitizens
   in Census Data. https://www.migrationpolicy.org/about/mpi-methodology-assigning-legal-status-noncitizens-census-data

5. Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.

---

*This report was generated automatically. For questions or feedback, see the project repository.*
"""

    return report


def generate_admin_comparison_section(
    acs_estimates: pd.DataFrame,
    admin_totals: Optional[pd.DataFrame],
    year: int,
) -> str:
    """
    Generate markdown section comparing ACS estimates to administrative totals.

    Args:
        acs_estimates: ACS-based rate estimates
        admin_totals: Administrative data totals (if available)
        year: Analysis year

    Returns:
        Markdown string for the comparison section
    """
    if admin_totals is None or admin_totals.empty:
        return """
## Administrative Data Comparison

*Administrative data comparison not available for this analysis.*

Administrative totals from CalFresh, Medi-Cal, CalWORKs, and SSI programs
would be compared to ACS survey estimates to assess survey underreporting
and validate imputation results.
"""

    # Build comparison table
    rows = []
    for _, row in admin_totals.iterrows():
        program = row.get("program", "")
        admin_count = row.get("admin_total", 0)
        admin_month = row.get("reference_month", "")

        # Find corresponding ACS estimate
        acs_match = acs_estimates[acs_estimates["program"] == program]
        if not acs_match.empty:
            acs_total = acs_match["weighted_n"].sum() if "weighted_n" in acs_match.columns else "N/A"
            if isinstance(acs_total, (int, float)):
                ratio = acs_total / admin_count if admin_count > 0 else np.nan
                ratio_str = f"{ratio:.2f}" if pd.notna(ratio) else "N/A"
                acs_str = f"{acs_total:,.0f}"
            else:
                ratio_str = "N/A"
                acs_str = str(acs_total)
        else:
            acs_str = "N/A"
            ratio_str = "N/A"

        rows.append(f"| {program} | {admin_count:,.0f} | {acs_str} | {ratio_str} | {admin_month} |")

    table = "\n".join(rows)

    section = f"""
## Administrative Data Comparison

The table below compares ACS survey-based counts to administrative data totals
from California state agencies. Ratios below 1.0 indicate survey underreporting.

| Program | Admin Total | ACS Estimate | Ratio | Admin Reference |
|---------|-------------|--------------|-------|-----------------|
{table}

### Interpretation Notes

1. **Definitional Differences:** Administrative data typically reflects a point-in-time
   count (monthly), while ACS asks about receipt in the past 12 months. This creates
   inherent differences even without underreporting.

2. **Survey Underreporting:** Research consistently finds that ACS captures approximately:
   - 60-70% of administrative SNAP/CalFresh recipients
   - 80-90% of Medicaid/Medi-Cal recipients
   - 70-80% of SSI recipients

3. **Coverage Differences:** Administrative data includes all recipients regardless of
   survey coverage, while ACS covers the household population (excluding group quarters
   in many cases).
"""
    return section


def generate_variance_decomposition_section(
    results: pd.DataFrame,
    bootstrap_results: Optional[dict] = None,
) -> str:
    """
    Generate markdown section showing variance decomposition.

    Args:
        results: Rate estimation results with variance components
        bootstrap_results: Optional bootstrap results with variance decomposition

    Returns:
        Markdown string for the variance decomposition section
    """
    # Check if variance components are available
    has_var_components = all(
        col in results.columns
        for col in ["se", "within_var", "between_var"]
    )

    if not has_var_components and bootstrap_results is None:
        return """
## Variance Decomposition

*Detailed variance decomposition not available for this analysis.*

When available, this section shows the contribution of different uncertainty
sources to total variance:
- **Survey sampling variance** (from replicate weights)
- **Imputation variance** (between-imputation variation)
- **Model uncertainty** (from bootstrap model training)
"""

    # Build variance decomposition table
    rows = []

    if has_var_components:
        for _, row in results.iterrows():
            program = row.get("program", "")
            group = row.get("group", "")
            total_var = row.get("se", 0) ** 2 if row.get("se", 0) > 0 else 0
            within_var = row.get("within_var", 0)
            between_var = row.get("between_var", 0)

            if total_var > 0:
                survey_pct = (within_var / total_var * 100) if within_var else 0
                mi_pct = (between_var / total_var * 100) if between_var else 0
                # Model variance (if bootstrap)
                model_var = row.get("model_var", 0)
                model_pct = (model_var / total_var * 100) if model_var else 0

                rows.append(
                    f"| {program} | {group} | {row.get('se', 0)*100:.2f}pp | "
                    f"{survey_pct:.1f}% | {mi_pct:.1f}% | {model_pct:.1f}% |"
                )

    if bootstrap_results:
        # Add bootstrap-based decomposition
        for key, result in bootstrap_results.items():
            if hasattr(result, "fraction_survey"):
                rows.append(
                    f"| {key} | Bootstrap | {result.se_total*100:.2f}pp | "
                    f"{result.fraction_survey*100:.1f}% | {result.fraction_mi*100:.1f}% | "
                    f"{result.fraction_model*100:.1f}% |"
                )

    if not rows:
        return """
## Variance Decomposition

*Variance components could not be computed for this analysis.*
"""

    table = "\n".join(rows)

    section = f"""
## Variance Decomposition

The table below shows the contribution of different uncertainty sources to
total standard error. Understanding variance composition helps identify which
aspects of the methodology contribute most to estimate uncertainty.

| Program | Group | Total SE | Survey % | MI % | Model % |
|---------|-------|----------|----------|------|---------|
{table}

### Component Definitions

- **Survey %**: Variance from survey sampling, estimated via successive difference
  replication (SDR) with 80 replicate weights.

- **MI %**: Variance from multiple imputation, reflecting uncertainty in assigning
  legal status to individual noncitizens.

- **Model %**: Variance from model estimation (bootstrap), capturing uncertainty
  in the SIPP-trained classification model parameters.

### Interpretation

- When **Survey %** dominates: Sample size is the limiting factor. Pooling years
  or using smaller geographic units may help.

- When **MI %** dominates: Status imputation uncertainty is high. This typically
  occurs for undocumented-specific estimates.

- When **Model %** dominates: The classification model has high parameter uncertainty.
  This may indicate insufficient SIPP training data.

**Formula:** $V_{{total}} = V_{{survey}} + (1 + 1/M) \\times V_{{MI}} + (1 + 1/B) \\times V_{{model}}$

Where M = number of imputations, B = number of bootstrap replicates.
"""
    return section


def generate_enhanced_limitations_section(
    has_admin_data: bool = False,
    has_bootstrap: bool = False,
    has_pooled: bool = False,
) -> str:
    """
    Generate enhanced limitations section with context-specific caveats.

    Args:
        has_admin_data: Whether admin data comparison was performed
        has_bootstrap: Whether bootstrap model uncertainty was estimated
        has_pooled: Whether pooled multi-year analysis was performed

    Returns:
        Markdown string for the limitations section
    """
    base_limitations = """
## Limitations and Caveats

### Core Methodological Limitations

1. **Imputation Uncertainty:** Legal status is imputed, not observed. Estimates for
   undocumented immigrants have substantially higher uncertainty than for other groups.
   The imputation model relies on observable covariates that correlate with, but do
   not deterministically identify, legal status.

2. **Survey Underreporting:** Survey-based benefit measures systematically underestimate
   true program receipt. Research indicates ACS SNAP receipt captures approximately
   60-70% of administrative totals. Underreporting may vary by immigration status,
   potentially biasing comparative rates.

3. **Eligibility vs. Receipt:** This analysis measures program receipt, not eligibility.
   Many eligible individuals do not participate in programs due to:
   - Lack of awareness
   - Administrative barriers
   - Fear of public charge implications (for immigrants)
   - Stigma or personal preference

4. **Household Complexity:** Mixed-status households (containing both documented and
   undocumented members) complicate individual-level classification. Program eligibility
   and receipt often depend on household composition, not just individual status.

5. **Temporal Limitations:** Results reflect the survey reference period and may not
   represent current policy conditions or economic circumstances.

6. **Small Sample Sizes:** Subgroup estimates (especially for undocumented by detailed
   demographic characteristics) may have large sampling error, leading to unreliable
   estimates flagged with high coefficients of variation.

7. **Model Transfer Assumptions:** The SIPP-to-ACS model transfer assumes that
   relationships between covariates and legal status are stable across surveys.
   Differences in survey design, coverage, and timing may introduce bias.
"""

    conditional_limitations = ""

    if not has_admin_data:
        conditional_limitations += """
### Administrative Data Limitations

This analysis does not incorporate administrative data validation. Without
comparison to administrative totals, we cannot directly assess:
- Magnitude of survey underreporting by program
- Differential underreporting by immigration status
- Temporal alignment of estimates with actual program caseloads
"""

    if not has_bootstrap:
        conditional_limitations += """
### Model Uncertainty Limitations

This analysis uses a single trained imputation model. Without bootstrap
replication of model training, the variance estimates do not fully capture:
- Uncertainty in model coefficients
- Sensitivity to SIPP training sample composition
- Potential overfitting to specific SIPP panel characteristics

Full variance decomposition requires bootstrap model uncertainty estimation.
"""

    if has_pooled:
        conditional_limitations += """
### Pooled Analysis Considerations

Multi-year pooling increases effective sample size but introduces:
- Assumption of stable covariate-outcome relationships across years
- Potential masking of year-to-year policy changes
- Complexity in interpreting time-varying eligibility rules
- Weight adjustments that may not fully account for temporal correlation
"""

    return base_limitations + conditional_limitations


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS year (default: 2023)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Report Generation - {args.year}")
    logger.info("=" * 60)

    # Create output directory
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    try:
        # Try imputed first
        main_results = load_results(args.year, suffix="_imputed")
        source = "imputed"
    except FileNotFoundError:
        try:
            main_results = load_results(args.year)
            source = "combined"
        except FileNotFoundError:
            try:
                main_results = load_results(args.year, suffix="_observable")
                source = "observable"
            except FileNotFoundError:
                logger.error("No results files found")
                logger.error("Run 'python -m src.04_estimate_rates' first")
                return 1

    logger.info(f"Using {source} results")

    # Generate figures
    figure_paths = []

    try:
        bar_chart = create_bar_chart(main_results, args.year, config.FIGURES_DIR)
        figure_paths.append(bar_chart)
    except KeyError as e:
        logger.warning(f"Missing data column for bar chart: {e}")
    except ValueError as e:
        logger.warning(f"Invalid data values for bar chart: {e}")
    except OSError as e:
        logger.warning(f"Could not save bar chart: {e}")

    # Try to load observable for comparison
    try:
        observable_results = load_results(args.year, suffix="_observable")
        if source == "imputed":
            comparison = create_comparison_chart(
                observable_results, main_results, args.year, config.FIGURES_DIR
            )
            figure_paths.append(comparison)
    except FileNotFoundError as e:
        logger.warning(
            f"Observable results not found for comparison ({e}). "
            "Skipping comparison chart. Run with --observable-only first if comparison is needed."
        )

    # Try to load admin data for validation
    admin_totals = None
    try:
        admin_path = config.ADMIN_DATA_DIR / f"admin_totals_{args.year}.csv"
        if admin_path.exists():
            admin_totals = pd.read_csv(admin_path)
            logger.info(f"Loaded admin totals from {admin_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.info(f"Admin totals not available: {e}")

    # Generate report with validation sections
    report = generate_report_markdown(
        main_results,
        args.year,
        figure_paths,
        admin_totals=admin_totals,
        bootstrap_results=None,  # Will be populated when bootstrap is run
        is_pooled=False,
    )

    # Save report
    report_path = config.REPORTS_DIR / f"ca_welfare_by_immigration_status_{args.year}.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved report: {report_path}")

    # Also save as the main report file
    main_report_path = config.REPORTS_DIR / config.OUTPUT_REPORT_FILE
    with open(main_report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved main report: {main_report_path}")

    logger.info("\nReport generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
