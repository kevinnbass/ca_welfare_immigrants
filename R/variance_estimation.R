# Variance estimation using R's survey package
#
# This script provides proper variance estimation using ACS replicate weights
# via successive difference replication (SDR).
#
# Usage:
#   Rscript R/variance_estimation.R <input_parquet> <output_csv>

library(survey)
library(srvyr)
library(arrow)
library(tidyverse)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript variance_estimation.R <input_parquet> <output_csv>\n")
  quit(status = 1)
}

input_file <- args[1]
output_file <- args[2]

cat("Loading data from:", input_file, "\n")

# Read parquet file
df <- read_parquet(input_file)

cat("Loaded", nrow(df), "records\n")

# Check for replicate weights
rep_cols <- paste0("PWGTP", 1:80)
has_rep_weights <- all(rep_cols %in% names(df))

if (!has_rep_weights) {
  cat("WARNING: Replicate weights not found. Using simple weighted estimates.\n")
}

# Define the survey design
# ACS uses successive difference replication (SDR)
# The multiplier is 4/80 for SDR

if (has_rep_weights) {
  # Create survey design with replicate weights
  svy_design <- df %>%
    as_survey_rep(
      weights = PWGTP,
      repweights = starts_with("PWGTP"),
      type = "successive-difference",
      mse = TRUE,
      combined_weights = TRUE
    )
} else {
  # Simple design (no variance estimation)
  svy_design <- df %>%
    as_survey_design(weights = PWGTP)
}

# Function to compute rates by group
compute_rates <- function(design, indicator_col, status_col) {

  results <- design %>%
    group_by(!!sym(status_col)) %>%
    summarise(
      estimate = survey_mean(!!sym(indicator_col), na.rm = TRUE, vartype = c("se", "ci")),
      n_unweighted = unweighted(n()),
      n_weighted = survey_total(1)
    ) %>%
    rename(
      group = !!sym(status_col),
      rate = estimate,
      se = estimate_se,
      ci_lower = estimate_low,
      ci_upper = estimate_upp,
      pop_total = n_weighted
    )

  return(results)
}

# Programs to analyze
programs <- c("medicaid", "snap", "ssi", "public_assistance", "any_benefit")
programs <- programs[programs %in% names(df)]

# Status column (use observable if imputed not available)
if ("status_agg_0" %in% names(df)) {
  status_col <- "status_agg_0"
  cat("Using imputed status (imputation 0)\n")
} else if ("observable_status" %in% names(df)) {
  status_col <- "observable_status"
  cat("Using observable status\n")
} else {
  cat("ERROR: No status column found\n")
  quit(status = 1)
}

# Compute rates for each program
all_results <- data.frame()

for (program in programs) {
  cat("Computing rates for:", program, "\n")

  tryCatch({
    result <- compute_rates(svy_design, program, status_col)
    result$program <- program
    all_results <- bind_rows(all_results, result)
  }, error = function(e) {
    cat("  Error:", e$message, "\n")
  })
}

# Add CV column
all_results <- all_results %>%
  mutate(cv = se / rate)

# Save results
cat("Saving results to:", output_file, "\n")
write_csv(all_results, output_file)

cat("Done!\n")
