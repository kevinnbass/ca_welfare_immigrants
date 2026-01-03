# Multiple imputation combining using Rubin's rules
#
# This script combines estimates across multiple imputations using
# the mitools package for proper variance estimation.
#
# Usage:
#   Rscript R/mi_combine.R <input_parquet> <n_imputations> <output_csv>

library(mitools)
library(survey)
library(srvyr)
library(arrow)
library(tidyverse)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript mi_combine.R <input_parquet> <n_imputations> <output_csv>\n")
  quit(status = 1)
}

input_file <- args[1]
n_imputations <- as.integer(args[2])
output_file <- args[3]

cat("Loading data from:", input_file, "\n")
cat("Number of imputations:", n_imputations, "\n")

# Read parquet file
df <- read_parquet(input_file)

cat("Loaded", nrow(df), "records\n")

# Check for replicate weights
rep_cols <- paste0("PWGTP", 1:80)
has_rep_weights <- all(rep_cols %in% names(df))

# Create list of imputed datasets/designs
imputed_designs <- list()

for (i in 0:(n_imputations - 1)) {
  status_col <- paste0("status_agg_", i)
  weight_col <- paste0("weight_cal_", i)

  if (!status_col %in% names(df)) {
    cat("WARNING: Missing status column:", status_col, "\n")
    next
  }

  # Use calibrated weights if available, otherwise use PWGTP
  if (!weight_col %in% names(df)) {
    weight_col <- "PWGTP"
  }

  # Create design for this imputation
  if (has_rep_weights) {
    design <- df %>%
      mutate(status = !!sym(status_col)) %>%
      as_survey_rep(
        weights = !!sym(weight_col),
        repweights = starts_with("PWGTP"),
        type = "successive-difference",
        mse = TRUE
      )
  } else {
    design <- df %>%
      mutate(status = !!sym(status_col)) %>%
      as_survey_design(weights = !!sym(weight_col))
  }

  imputed_designs[[i + 1]] <- design
}

cat("Created", length(imputed_designs), "survey designs\n")

# Programs to analyze
programs <- c("medicaid", "snap", "ssi", "public_assistance", "any_benefit")
programs <- programs[programs %in% names(df)]

# Status groups
status_groups <- c("US_BORN", "LEGAL_IMMIGRANT", "UNDOCUMENTED")

# Combine results using mitools approach
# For each group and program, compute estimate from each imputation
# then combine using Rubin's rules

all_results <- data.frame()

for (program in programs) {
  cat("Processing:", program, "\n")

  for (group in status_groups) {

    estimates <- numeric(length(imputed_designs))
    variances <- numeric(length(imputed_designs))

    for (i in seq_along(imputed_designs)) {
      design <- imputed_designs[[i]]

      # Filter to group
      group_design <- design %>%
        filter(status == group)

      # Compute mean
      tryCatch({
        result <- group_design %>%
          summarise(
            est = survey_mean(!!sym(program), na.rm = TRUE, vartype = "var")
          )
        estimates[i] <- result$est
        variances[i] <- result$est_var
      }, error = function(e) {
        estimates[i] <- NA
        variances[i] <- NA
      })
    }

    # Apply Rubin's rules
    m <- length(estimates)
    q_bar <- mean(estimates, na.rm = TRUE)
    u_bar <- mean(variances, na.rm = TRUE)
    b <- var(estimates, na.rm = TRUE)

    # Total variance
    total_var <- u_bar + (1 + 1/m) * b
    se <- sqrt(total_var)

    # Degrees of freedom (Barnard-Rubin)
    r <- (1 + 1/m) * b / u_bar
    df <- (m - 1) * (1 + 1/r)^2

    # CI using t-distribution
    t_crit <- qt(0.975, df)
    ci_lower <- q_bar - t_crit * se
    ci_upper <- q_bar + t_crit * se

    # Fraction of missing information
    fmi <- (1 + 1/m) * b / total_var

    # Get n from first imputation
    n_unweighted <- imputed_designs[[1]] %>%
      filter(status == group) %>%
      summarise(n = unweighted(n())) %>%
      pull(n)

    all_results <- bind_rows(all_results, data.frame(
      program = program,
      group = group,
      estimate = q_bar,
      se = se,
      ci_lower = ci_lower,
      ci_upper = ci_upper,
      within_var = u_bar,
      between_var = b,
      total_var = total_var,
      df = df,
      fmi = fmi,
      n_imputations = m,
      n_unweighted = n_unweighted
    ))
  }
}

# Add CV and reliability flags
all_results <- all_results %>%
  mutate(
    cv = se / estimate,
    reliable = (n_unweighted >= 30) & (cv <= 0.30),
    suppressed = n_unweighted < 30
  )

# Save results
cat("Saving results to:", output_file, "\n")
write_csv(all_results, output_file)

cat("Done!\n")
