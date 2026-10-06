# Independent R reader; no production analysis or GPU execution.
# logp_field() is loaded by the CLI from compare_logp.R in the same directory.
compare_nonp_objects <- function(reference,result) {
  atol <- 1e-10; rtol <- 1e-7
  schema_errors <- 0L; numeric_fields <- 0L; numeric_cells <- 0
  numeric_failed_cells <- 0; skipped_p_cells <- 0; numeric_errors <- 0L
  maximum_absolute_difference <- 0; maximum_relative_difference <- 0
  p_field <- function(name) !is.na(name)&(logp_field(name)|name%in%c('pvalue_log','pvalue_log10'))
  schema_error <- function() schema_errors <<- schema_errors+1L
  visit <- function(a,b,skip_p=FALSE) {
    if(!identical(typeof(a),typeof(b)) || length(a)!=length(b) || !identical(isS4(a),isS4(b))) {
      schema_error();return(invisible(NULL))
    }
    aa <- attributes(a);bb <- attributes(b)
    if(!identical(names(aa),names(bb)))schema_error()
    checked <- intersect(names(aa),names(bb))
    # S4 slots carry values and are recursively checked below. Class/package
    # metadata is exact; treating numeric slots as structural attributes would
    # incorrectly replace their strict abs/relative contract with bit equality.
    if(isS4(a))checked <- intersect(checked,'class')
    for(attribute in checked)if(!identical(aa[[attribute]],bb[[attribute]]))schema_error()
    if(isS4(a)) {
      if(!identical(slotNames(a),slotNames(b)))schema_error()
      for(name in intersect(slotNames(a),slotNames(b)))visit(slot(a,name),slot(b,name),skip_p)
      valid <- tryCatch(validObject(b),error=function(e)FALSE)
      if(!isTRUE(valid))schema_error()
    } else if(is.list(a)) {
      columns <- if(is.matrix(a))colnames(a)else names(a)
      for(i in seq_along(a)) {
        j <- if(is.matrix(a) && nrow(a)>0L)1L+(i-1L)%/%nrow(a)else i
        selected <- skip_p || (!is.null(columns) && p_field(columns[j]))
        visit(a[[i]],b[[i]],selected)
      }
    } else if(is.double(a)||is.complex(a)) {
      # P-valued numeric leaves still obey the exact missing/nonfinite schema.
      if(!identical(is.na(a),is.na(b)))schema_error()
      if(!identical(is.nan(a),is.nan(b)))schema_error()
      if(!identical(is.finite(a),is.finite(b)))schema_error()
      infinity <- is.infinite(a)&is.infinite(b)
      if(any(infinity)&&!identical(a[infinity],b[infinity]))schema_error()
      skip <- rep(skip_p,length(a))
      if(is.matrix(a)&&!is.null(colnames(a))) {
        skip <- skip | rep(p_field(colnames(a)),each=nrow(a))
      } else if(!is.null(names(a)))skip <- skip | p_field(names(a))
      skipped_p_cells <<- skipped_p_cells+sum(skip)
      numeric_cells <<- numeric_cells+sum(!skip)
      if(any(!skip))numeric_fields <<- numeric_fields+1L
      finite <- !skip&is.finite(a)&is.finite(b)
      if(any(finite)) {
        d <- abs(a[finite]-b[finite]);relative <- d/abs(a[finite]);relative[d==0] <- 0
        maximum_absolute_difference <<- max(maximum_absolute_difference,d)
        maximum_relative_difference <<- max(maximum_relative_difference,relative)
        failed <- d>atol+rtol*abs(a[finite])
        numeric_failed_cells <<- numeric_failed_cells+sum(failed)
        if(any(failed))numeric_errors <<- numeric_errors+1L
      }
    } else if(!identical(a,b))schema_error()
    invisible(NULL)
  }
  visit(reference,result)
  list(schema_version=1L,atol=atol,rtol=rtol,schema_errors=schema_errors,
    numeric_fields=numeric_fields,numeric_cells=numeric_cells,
    numeric_failed_cells=numeric_failed_cells,numeric_errors=numeric_errors,
    skipped_p_cells=skipped_p_cells,maximum_absolute_difference=maximum_absolute_difference,
    maximum_relative_difference=if(is.finite(maximum_relative_difference))maximum_relative_difference else 'Inf',
    passed=schema_errors==0L&&numeric_errors==0L,
    scope='All schema and non-P strict numeric comparison; P numeric error judged separately by logP')
}
if(sys.nframe()==0L) {
  args <- commandArgs(TRUE)
  if(length(args)!=3L)stop('Rscript validation/compare_nonp.R reference.Rdata candidate.Rdata report.json')
  file_arg <- grep('^--file=',commandArgs(FALSE),value=TRUE)
  script <- sub('^--file=','',file_arg[1L])
  source(file.path(dirname(script),'compare_logp.R'))
  suppressPackageStartupMessages(library(Matrix))
  suppressPackageStartupMessages(library(jsonlite))
  report <- compare_nonp_objects(read_logp_native(args[1L]),read_logp_native(args[2L]))
  write_json(report,args[3L],auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
  cat('nonp_cells:',report$numeric_cells,'nonp_passed:',report$passed,'\n')
  if(!report$passed)quit(status=1L)
}
