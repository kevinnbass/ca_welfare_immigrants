# Install required R packages for survey variance estimation
# Run this script once before using the R-based variance estimation

# Check if required packages are installed, install if not
required_packages <- c(
  "survey",      # Complex survey analysis with replicate weights
  "srvyr",       # Tidyverse-friendly wrapper for survey package
  "mitools",     # Tools for multiple imputation analysis
  "haven",       # Read SAS/Stata/SPSS files
  "tidyverse",   # Data manipulation and visualization
  "arrow"        # Read parquet files for Python-R data exchange
)

# Function to install missing packages
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(paste("Installing package:", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(paste("Package already installed:", pkg))
  }
}

# Install all required packages
message("Checking and installing required R packages...")
invisible(sapply(required_packages, install_if_missing))

# Verify installations
message("\nVerifying package installations:")
for (pkg in required_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(paste("  [OK]", pkg, packageVersion(pkg)))
  } else {
    message(paste("  [FAILED]", pkg))
  }
}

message("\nR package installation complete.")
message("You can now run the variance estimation scripts.")
