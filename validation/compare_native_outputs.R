# Independent oracle check only. Production PyTorch analysis never runs R.
# Usage: Rscript validation/compare_native_outputs.R expected.Rdata actual.Rdata
# Optional third/fourth arguments: absolute / relative numeric tolerance.
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2L) stop("supply expected and actual native R files")
atol <- if (length(args) > 2L) as.double(args[3L]) else 1e-10
rtol <- if (length(args) > 3L) as.double(args[4L]) else 1e-7
read_native <- function(path) {
  if (tolower(tools::file_ext(path)) == "rds") return(list(RDS=readRDS(path)))
  environment <- new.env(parent=baseenv())
  saved_names <- load(path, envir=environment)
  mget(saved_names, envir=environment, inherits=FALSE)
}
expected <- read_native(args[1L]); actual <- read_native(args[2L])
schema_errors <- character(); numeric_errors <- character()
numeric_fields <- 0L; numeric_cells <- 0; numeric_failed_cells <- 0
maximum_absolute_difference <- 0; maximum_relative_difference <- 0
schema_error <- function(path, reason) {
  schema_errors <<- c(schema_errors, paste(path, reason, sep=": "))
}
compare <- function(reference, result, path, inspect_attributes=TRUE) {
  if (!identical(typeof(reference), typeof(result))) {
    schema_error(path, "typeof differs"); return(invisible(NULL))
  }
  if (length(reference) != length(result)) {
    schema_error(path, "length differs"); return(invisible(NULL))
  }
  if (inspect_attributes) {
    a <- attributes(reference); b <- attributes(result)
    if (!identical(names(a), names(b))) schema_error(path, "attribute names/order differ")
    checked_attributes <- intersect(names(a), names(b))
    if (isS4(reference)) checked_attributes <- intersect(checked_attributes, "class")
    for (attribute in checked_attributes) {
      # Shape, names, class and factor levels must be exact, independently of
      # the tolerance used for numeric association results.
      if (!identical(a[[attribute]], b[[attribute]]))
        schema_error(paste0(path, "@", attribute), "attribute differs")
    }
  }
  if (isS4(reference)) {
    if (!identical(slotNames(reference), slotNames(result))) schema_error(path, "S4 slots differ")
    for (slot in intersect(slotNames(reference), slotNames(result)))
      compare(slot(reference, slot), slot(result, slot), paste0(path, "@", slot))
    valid <- tryCatch(validObject(result), error=function(error) FALSE)
    if (!isTRUE(valid)) schema_error(path, "invalid S4 object")
  } else if (is.list(reference)) {
    for (index in seq_along(reference))
      compare(reference[[index]], result[[index]], paste0(path, "[[", index, "]]"))
  } else if (is.double(reference) || is.complex(reference)) {
    numeric_fields <<- numeric_fields + 1L
    numeric_cells <<- numeric_cells + length(reference)
    if (!identical(is.na(reference), is.na(result))) schema_error(path, "missing values differ")
    if (!identical(is.nan(reference), is.nan(result))) schema_error(path, "NA and NaN differ")
    finite <- is.finite(reference) & is.finite(result)
    if (!identical(is.finite(reference), is.finite(result))) schema_error(path, "nonfinite values differ")
    infinity <- is.infinite(reference) & is.infinite(result)
    if (any(infinity) && !identical(reference[infinity], result[infinity]))
      schema_error(path, "infinite values differ")
    if (any(finite)) {
      difference <- abs(reference[finite] - result[finite])
      maximum_absolute_difference <<- max(maximum_absolute_difference, difference)
      relative <- difference / abs(reference[finite])
      relative[difference == 0] <- 0
      maximum_relative_difference <<- max(maximum_relative_difference, relative)
      failed <- difference > atol + rtol * abs(reference[finite])
      numeric_failed_cells <<- numeric_failed_cells + sum(failed)
      if (any(failed))
        numeric_errors <<- c(numeric_errors, path)
    }
  } else if (!identical(reference, result)) {
    schema_error(path, "value differs")
  }
  invisible(NULL)
}
compare(expected, actual, "saved_objects")
cat("schema_errors:", length(schema_errors), "\n")
cat("numeric_errors:", length(numeric_errors), "\n")
cat("numeric_fields:", numeric_fields, "\n")
cat("numeric_cells:", format(numeric_cells, scientific=FALSE), "\n")
cat("numeric_failed_cells:", format(numeric_failed_cells, scientific=FALSE), "\n")
cat("maximum_absolute_difference:", format(maximum_absolute_difference, digits=17), "\n")
cat("maximum_relative_difference:", format(maximum_relative_difference, digits=17), "\n")
if (length(schema_errors)) cat(paste(schema_errors, collapse="\n"), "\n")
if (length(numeric_errors)) cat(paste(numeric_errors, collapse="\n"), "\n")
if (length(schema_errors) || length(numeric_errors)) quit(status=1L)
