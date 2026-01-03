# California Welfare Participation by Immigration Status

## Project Overview

This project estimates welfare program participation rates in California across three immigration status groups:
- **US-Born**: Persons born in the United States
- **Legal Immigrants**: Naturalized citizens and lawfully present noncitizens
- **Illegal Immigrants**: Foreign-born noncitizens without legal status (statistically imputed)

The analysis uses publicly available Census data (ACS PUMS and SIPP) with advanced statistical methods for status imputation and uncertainty quantification.

## Architecture

### Pipeline Flow

```
00_fetch_data.py     -> Download ACS, SIPP, Pew benchmark data
01_clean_acs.py      -> Clean ACS, create welfare indicators
02_train_status_model.py -> Train SIPP-based status classification model
03_impute_status_acs.py  -> Apply model to impute legal status in ACS
04_estimate_rates.py     -> Compute rates with uncertainty (Rubin's rules)
05_report.py             -> Generate visualizations and report
run_all.py               -> Orchestrate full pipeline
```

### Key Directories

- `src/` - Main Python modules
- `src/utils/` - Utility modules (download, weights, imputation, validation)
- `data/raw/` - Downloaded raw data (gitignored)
- `data/processed/` - Cleaned/intermediate data
- `data/external/` - External benchmarks (Pew)
- `outputs/tables/` - Result CSV files
- `outputs/figures/` - Generated visualizations
- `reports/` - Markdown reports
- `models/` - Trained model files
- `tests/` - Test suite

## Development Guidelines

### Error Handling

**Always use specific exception types:**
```python
# Good
except requests.HTTPError as e:
    logger.error(f"HTTP error: {e}")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")

# Avoid
except Exception as e:
    logger.error(f"Error: {e}")
```

### Validation Patterns

The codebase uses **strict validation mode** - functions raise errors rather than silently continuing with invalid input.

**Validate columns before DataFrame operations:**
```python
from . import config

config.validate_required_columns(df, ["col1", "col2"], context="function_name")
```

**Validate year arguments:**
```python
config.validate_year(args.year)  # Raises ValueError if outside 2015-2030
```

**Validate probability bounds (enforced in imputation functions):**
```python
# create_bernoulli_imputations() automatically validates:
# - n_imputations >= 1
# - No NaN values in probabilities
# - All probabilities in [0, 1]
```

**Validate calibration parameters:**
```python
# calibrate_to_total() raises ValueError if:
# - target_total <= 0
# - current_total is effectively zero
# - calibration ratio > 100 or < 0.01 (extreme adjustment)
```

**Validate binary indicators:**
```python
# weighted_proportion() validates indicator is binary (0/1) by default
weighted_proportion(indicator, weights, validate_binary=True)
```

**Validate replicate weights:**
```python
# Rate estimation requires exactly 80 replicate weights
# Raises ValueError if missing or incomplete
```

### Type Safety

Use the `StatusGroup` enum for immigration status values:
```python
from .config import StatusGroup

status = StatusGroup.US_BORN  # Instead of "US_BORN" string literal
```

### Configuration

All configurable values should be in `src/config.py`:
- `AGE_BINS`, `AGE_LABELS` - Demographic binning
- `MIN_UNWEIGHTED_N`, `MAX_COEFFICIENT_OF_VARIATION` - Statistical thresholds
- `VALID_YEAR_RANGE` - Acceptable data years
- `StatusGroup` enum - Immigration status categories (use this instead of string literals)
- `ensure_directories()` - Call explicitly to create required directories

**Directory initialization:**
```python
from . import config

# Call at pipeline entry points to ensure directories exist
config.ensure_directories()
```

### Logging

Use structured logging with appropriate levels:
```python
logger.info("Starting operation")
logger.debug("Detailed debugging info")
logger.warning("Non-fatal issue")
logger.error("Critical error")
```

## Running the Pipeline

### Full Pipeline
```bash
python -m src.run_all --year 2023
```

### Individual Steps
```bash
python -m src.00_fetch_data --year 2023
python -m src.01_clean_acs --year 2023 --validate
python -m src.02_train_status_model --model both
python -m src.03_impute_status_acs --year 2023
python -m src.04_estimate_rates --year 2023
python -m src.05_report --year 2023
```

### Observable-Only Mode (skip imputation)
```bash
python -m src.run_all --year 2023 --observable-only
```

## Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
pytest tests/unit/              # Unit tests only
pytest -m "not slow"            # Skip slow tests
pytest -m integration           # Integration tests only
pytest tests/unit/test_download.py  # Specific file
```

### Key Test Files
- `tests/unit/test_download.py` - Path traversal protection, retry logic
- `tests/unit/test_weights.py` - Weighted calculations, SDR variance, binary validation
- `tests/unit/test_imputation.py` - Rubin's rules, Bernoulli imputation, calibration
- `tests/unit/test_validation.py` - Data validation, population checks, CV thresholds
- `tests/unit/test_config.py` - Configuration validation, StatusGroup enum

## Known Limitations

1. **Imputation Uncertainty**: Legal status is imputed, not observed. Illegal immigrant estimates have higher uncertainty.

2. **Survey Underreporting**: ACS SNAP receipt captures ~60-70% of administrative totals.

3. **Eligibility vs Receipt**: Analysis measures program receipt, not eligibility.

4. **Mixed-Status Households**: Households with both legal and illegal members complicate classification.

5. **Model Transfer**: Assumes SIPP-to-ACS covariate relationships are stable.

## Security Considerations

- ZIP extraction includes path traversal protection
- No hardcoded credentials
- External URLs validated before download
- Year inputs validated against acceptable range

## Dependencies

### Required
- Python 3.11+
- pandas, numpy, scikit-learn, statsmodels
- requests, tqdm, pyyaml, joblib

### Optional
- R 4.0+ with survey package (for advanced variance estimation)

## Data Sources

1. **ACS PUMS** - Census Bureau microdata
2. **SIPP** - For status imputation model training
3. **Pew Research** - Calibration benchmarks for unauthorized population
