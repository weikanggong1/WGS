# Significant-union precision gate; the all-P validator retains its diagnostic semantics.
# CLI: original_compare_logp.R reference candidate report.json [.001] [.05]
significant_logp_function <- function(original_validator) {
  oracle <- new.env(parent=globalenv())
  source(original_validator,local=oracle)
  stopifnot(is.function(oracle$compare_logp_objects),is.function(oracle$read_logp_native))
  original <- oracle$compare_logp_objects
  expressions <- as.list(body(original))
  stopifnot(identical(expressions[[1L]],as.name("{")),length(expressions)>2L)
  # Execute the unchanged recursive extraction/validity/statistics body, then
  # extend its final report inside the SAME invocation. This avoids a second
  # independently guessed list of P field names or a different cell ordering.
  final_report <- expressions[[length(expressions)]]
  extension <- quote({
    report <- ORIGINAL_FINAL_REPORT
    significance_threshold <- .05
    reference_selected <- is.finite(a) & a>=0 & a<significance_threshold
    candidate_selected <- is.finite(b) & b>=0 & b<significance_threshold
    selected <- reference_selected | candidate_selected
    selected_summary <- summary(selected)
    selected_summary$selection <- "reference_P < 0.05 OR candidate_P < 0.05; strict threshold, union"
    selected_summary$p_threshold <- significance_threshold
    selected_summary$target_logp_absolute_difference <- target
    selected_summary$reference_selected_cells <- sum(reference_selected)
    selected_summary$candidate_selected_cells <- sum(candidate_selected)
    selected_summary$both_selected_cells <- sum(reference_selected & candidate_selected)
    selected_summary$reference_only_cells <- sum(reference_selected & !candidate_selected)
    selected_summary$candidate_only_cells <- sum(candidate_selected & !reference_selected)
    selected_summary$boundary_crossing_cells <- sum(xor(reference_selected,candidate_selected))
    selected_summary$reference_equal_threshold_cells <- sum(is.finite(a) & a==significance_threshold)
    selected_summary$candidate_equal_threshold_cells <- sum(is.finite(b) & b==significance_threshold)
    # Standalone native log fields retain the original validity checks. Their
    # precision is significant when either implied P is < threshold; central
    # stored-log differences remain in the unchanged all-P diagnostic report.
    stored_selected <- stored_ok & (stored_a > -log10(significance_threshold) |
                                   stored_b > -log10(significance_threshold))
    stored_sig_difference <- abs(stored_a[stored_selected]-stored_b[stored_selected])
    stored_sig <- list(total=sum(stored_selected),comparable=sum(stored_selected),uncomparable=0L,
      maximum_absolute_difference=if(length(stored_sig_difference))max(stored_sig_difference)else NULL,
      failed_cells=sum(stored_sig_difference>target))
    all_contract <- schema_errors==0L && length(a)>0 && all(comparable) &&
      all(stored_ok) && stored_consistency_errors==0L
    sig_pass <- all_contract && all(d[selected]<=target) && all(stored_sig_difference<=target)
    report$significant_gate_schema_version <- 1L
    report$all_P_contract_passed <- all_contract
    report$significant_logp <- selected_summary
    report$significant_stored_pvalue_log10 <- stored_sig
    report$significant_logp_passed <- sig_pass
    report$primary_acceptance_scope <- "All P cells valid/comparable and native logs consistent; absolute delta -log10(P)<=target for reference or candidate P<0.05 union; all-P precision diagnostic retained"
    report
  })
  # substitute binds only the final expression, leaving the original body and
  # its lexical helper functions untouched. Never alter original_validator.
  expressions[[length(expressions)]] <- substitute(EXTENSION,
      list(EXTENSION=do.call(substitute,list(extension,list(ORIGINAL_FINAL_REPORT=final_report)))))
  candidate <- original
  body(candidate) <- as.call(expressions)
  attr(candidate,"oracle") <- oracle
  candidate
}

if(sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  stopifnot(length(args)>=4L,length(args)<=6L)
  target <- if(length(args)>=5L)as.numeric(args[[5L]])else .001
  threshold <- if(length(args)>=6L)as.numeric(args[[6L]])else .05
  stopifnot(length(threshold)==1L,is.finite(threshold),identical(threshold,.05))
  compare <- significant_logp_function(args[[1L]])
  oracle <- attr(compare,"oracle")
  report <- compare(oracle$read_logp_native(args[[2L]]),oracle$read_logp_native(args[[3L]]),target)
  jsonlite::write_json(report,args[[4L]],auto_unbox=TRUE,pretty=TRUE,digits=17,null="null")
  cat("recognized_p_cells:",report$all$total,"significant_union_cells:",report$significant_logp$total,
      "significant_logp_passed:",report$significant_logp_passed,"allP_diagnostic_passed:",report$logp_passed,"\n")
  if(!report$significant_logp_passed)quit(status=1L)
}
