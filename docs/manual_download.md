# Manual Download Instructions

Some data sources require registration or manual download. This document provides step-by-step instructions.

## CHIS (California Health Interview Survey)

CHIS public use files require free registration with UCLA Center for Health Policy Research.

### Step 1: Create Account

1. Go to https://healthpolicy.ucla.edu/our-work/public-use-files
2. Click on "Request Access" or "Create Account"
3. Fill in required information (name, email, institution, intended use)
4. Agree to terms of use
5. Verify email and complete registration

### Step 2: Request Data

1. Log in to your account
2. Navigate to "One-Year PUFs" section
3. Select the most recent year available (2023 or 2024)
4. Select "Adult" file type
5. Submit request

### Step 3: Download and Place Files

1. Once approved, download the data files:
   - Adult PUF data file (SAS, Stata, or ASCII format)
   - Codebook/documentation
2. Place downloaded files in: `data/raw/chis/`
3. Rename files following this pattern:
   - Data: `chis_adult_YYYY.sas7bdat` (or appropriate extension)
   - Codebook: `chis_adult_YYYY_codebook.pdf`

### Step 4: Update Data Inventory

After downloading, add an entry to `docs/data_inventory.yaml`:

```yaml
download_log:
  - dataset: chis_adult_2023
    download_date: YYYY-MM-DD
    file_path: data/raw/chis/chis_adult_2023.sas7bdat
```

## Pew Research Data (If Direct Download Fails)

If the Pew Excel files cannot be downloaded automatically:

### Step 1: Navigate to Article

1. Go to https://www.pewresearch.org/race-and-ethnicity/
2. Search for "unauthorized immigrant population"
3. Find the most recent article with state-level data (August 2025 report)

### Step 2: Download Excel Files

1. Look for "Download detailed tables" or similar link
2. Download:
   - State trends Excel file
   - Labor force by state Excel file
3. Place in: `data/external/`

### Step 3: Rename Files

Rename to match expected names:
- `pew_state_trends.xlsx`
- `pew_labor_force_by_state.xlsx`

## MPI California Profile (If Needed)

The Migration Policy Institute data may need manual extraction:

### Step 1: Visit Profile Page

1. Go to https://www.migrationpolicy.org/data/unauthorized-immigrant-population/state/CA

### Step 2: Extract Data

1. Note the unauthorized population estimate for California
2. Note the year and confidence interval if provided
3. Create a file `data/external/mpi_ca_unauthorized.yaml`:

```yaml
source: Migration Policy Institute
url: https://www.migrationpolicy.org/data/unauthorized-immigrant-population/state/CA
access_date: YYYY-MM-DD
california:
  unauthorized_population: 2200000  # Update with actual value
  year: 2023
  notes: "Extracted manually from MPI website"
```

## Verification Checklist

After manual downloads, verify:

- [ ] CHIS adult file is in `data/raw/chis/`
- [ ] CHIS codebook is available for reference
- [ ] Pew Excel files are in `data/external/` (if manual download needed)
- [ ] MPI data is documented in `data/external/` (if manual extraction needed)
- [ ] Data inventory is updated with download dates

## Troubleshooting

### CHIS Registration Issues
- Check spam folder for verification email
- Contact: chis@ucla.edu

### Pew Download Issues
- Try different browser
- Check if URL has changed on Pew website
- Use Internet Archive (archive.org) if needed

### Data Format Issues
- Ensure you have appropriate software to read SAS/Stata files
- Python can read these with `pyreadstat` or `pandas` with appropriate engine
