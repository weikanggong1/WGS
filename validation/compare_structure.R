# Independent native R output verification; no production analysis or GPU.
# Floating-point values are deliberately not subject to FP64 matching.
compare_structure_objects <- function(reference, result) {
  errors <- 0L; floating_cells <- 0; exact_cells <- 0
  error <- function() errors <<- errors + 1L
  visit <- function(a, b) {
    if (!identical(typeof(a), typeof(b)) || length(a) != length(b) ||
        !identical(isS4(a), isS4(b))) {
      error(); return(invisible(NULL))
    }
    aa <- attributes(a); bb <- attributes(b)
    if (!identical(names(aa), names(bb))) error()
    checked <- intersect(names(aa), names(bb))
    # S4 numeric slots contain data, not attributes of the file format.
    if (isS4(a)) checked <- intersect(checked, 'class')
    for (name in checked) if (!identical(aa[[name]], bb[[name]])) error()
    if (isS4(a)) {
      if (!identical(slotNames(a), slotNames(b))) error()
      for (name in intersect(slotNames(a), slotNames(b)))
        visit(slot(a, name), slot(b, name))
      if (!isTRUE(tryCatch(validObject(b), error=function(e) FALSE))) error()
    } else if (is.list(a)) {
      for (i in seq_along(a)) visit(a[[i]], b[[i]])
    } else if (is.double(a) || is.complex(a)) {
      floating_cells <<- floating_cells + length(a)
      if (!identical(is.na(a), is.na(b))) error()
      if (!identical(is.nan(a), is.nan(b))) error()
      if (!identical(is.finite(a), is.finite(b))) error()
      infinity <- is.infinite(a) & is.infinite(b)
      if (any(infinity) && !identical(a[infinity], b[infinity])) error()
    } else {
      exact_cells <<- exact_cells + length(a)
      # Integer indices, factors, characters, logicals and NULL are exact.
      if (!identical(a, b)) error()
    }
    invisible(NULL)
  }
  visit(reference, result)
  list(schema_version=1L, schema_errors=errors, exact_cells=exact_cells,
       floating_cells=floating_cells, passed=errors==0L,
       scope='Native type, attributes, dimensions, ordering, identifiers, missing-value and NULL structure; no floating-point matching')
}
if (sys.nframe()==0L) {
  args <- commandArgs(TRUE)
  if (length(args)!=3L)
    stop('Rscript validation/compare_structure.R reference.Rdata candidate.Rdata report.json')
  file_arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
  script <- sub('^--file=', '', file_arg[1L])
  source(file.path(dirname(script), 'compare_logp.R'))
  suppressPackageStartupMessages(library(Matrix))
  suppressPackageStartupMessages(library(jsonlite))
  report <- compare_structure_objects(read_logp_native(args[1L]), read_logp_native(args[2L]))
  write_json(report, args[3L], auto_unbox=TRUE, pretty=TRUE, digits=17, null='null')
  cat('schema_errors:', report$schema_errors, 'structure_passed:', report$passed, '\n')
  if (!report$passed) quit(status=1L)
}
