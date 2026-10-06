# Independent R oracle validation; production analysis does not call R.
# Source this file for compare_logp_objects(), or use the CLI documented below.
logp_field <- function(name) {
  !is.na(name) & (name %in% c("pvalue", "Pvalue", "P.value", "p_value", "ACAT-O", "STAAR-O") |
    grepl("^(SKAT|Burden|ACAT-V)\\(1,(1|25)\\)(-[[:alnum:]_.()/-]+)?$", name) |
    grepl("^STAAR-[SBA]\\(1,(1|25)\\)$", name))
}
stable_negative_log10 <- function(p) {
  stopifnot(all(is.finite(p) & p > 0 & p <= 1))
  result <- numeric(length(p)); near_one <- p > .5
  result[near_one] <- -log1p(p[near_one]-1)/log(10)
  result[!near_one] <- -log10(p[!near_one])
  result
}
compare_logp_objects <- function(reference, result, target=.001) {
  stopifnot(length(target)==1L, is.finite(target), target >= 0)
  pairs <- list(); stored_pairs <- list(); schema_errors <- 0L; recognized_fields <- 0L
  add <- function(a,b,a_log=NULL,b_log=NULL) {
    if(!is.double(a) || !is.double(b) || length(a)!=length(b)) {
      schema_errors <<- schema_errors+1L; return(invisible(NULL))
    }
    recognized_fields <<- recognized_fields+1L
    pairs[[length(pairs)+1L]] <<- list(a=a,b=b,a_log=a_log,b_log=b_log)
  }
  visit <- function(a,b,p_field=FALSE) {
    if(!identical(typeof(a),typeof(b)) || length(a)!=length(b)) {
      schema_errors <<- schema_errors+1L; return(invisible(NULL))
    }
    if(!identical(attributes(a),attributes(b))) schema_errors <<- schema_errors+1L
    if(is.matrix(a)) {
      columns <- colnames(a)
      if(!is.null(columns)) for(j in seq_len(ncol(a))) {
        selected <- logp_field(columns[j])
        if(is.list(a)) for(i in seq_len(nrow(a))) {
          k <- i+(j-1L)*nrow(a); visit(a[[k]],b[[k]],selected)
        } else if(selected) add(a[,j],b[,j])
      }
      else if(is.list(a)) for(i in seq_along(a)) visit(a[[i]],b[[i]])
    } else if(is.list(a)) {
      labels <- names(a)
      for(i in seq_along(a)) {
        if(is.data.frame(a) && !is.null(labels) && labels[i]=="pvalue" && any(c("pvalue_log10","pvalue_log")%in%labels)) {
          native_name <- if("pvalue_log10"%in%labels)"pvalue_log10"else"pvalue_log"
          scale <- if(native_name=="pvalue_log")log(10)else 1
          add(a[[i]],b[[i]],a[[native_name]]/scale,b[[native_name]]/scale)
        } else visit(a[[i]],b[[i]], if(!is.null(labels) && labels[i]%in%c("pvalue_log10","pvalue_log")) if(labels[i]=="pvalue_log")"stored_ln"else"stored_log10" else !is.null(labels) && logp_field(labels[i]))
      }
    } else if(is.double(a) && !is.null(names(a)) && !isTRUE(p_field) &&
              !(p_field%in%c("stored_log10","stored_ln"))) {
      # Named atomic statistics vectors use names as field labels. Keep the
      # existing enclosing-vector attribute/type gate and value-NA contract.
      labels <- names(a)
      native_index <- which(labels=="pvalue_log10")
      native_scale <- 1
      if(!length(native_index)) {
        native_index <- which(labels=="pvalue_log"); native_scale <- log(10)
      }
      can_pair_native <- sum(labels=="pvalue",na.rm=TRUE)==1L && length(native_index)==1L
      for(i in seq_along(a)) {
        if(!is.na(labels[i]) && logp_field(labels[i])) {
          if(labels[i]=="pvalue" && can_pair_native)
            add(a[i],b[i],a[native_index]/native_scale,b[native_index]/native_scale)
          else add(a[i],b[i])
        } else if(!is.na(labels[i]) && labels[i]%in%c("pvalue_log10","pvalue_log")) {
          scale <- if(labels[i]=="pvalue_log")log(10)else 1
          stored_pairs[[length(stored_pairs)+1L]] <<- list(a=a[i]/scale,b=b[i]/scale)
        }
      }
    } else if(p_field%in%c("stored_log10","stored_ln")) {
      if(is.double(a) && is.double(b)) stored_pairs[[length(stored_pairs)+1L]] <<- list(a=a/if(identical(p_field,"stored_ln"))log(10)else 1,b=b/if(identical(p_field,"stored_ln"))log(10)else 1)
      else schema_errors <<- schema_errors+1L
    } else if(isTRUE(p_field)) add(a,b)
  }
  visit(reference,result)
  a <- as.double(unlist(lapply(pairs,function(x)x$a),use.names=FALSE))
  b <- as.double(unlist(lapply(pairs,function(x)x$b),use.names=FALSE))
  comparable <- is.finite(a)&is.finite(b)&a>0&a<=1&b>0&b<=1
  raw_comparable <- comparable
  a_log <- as.double(unlist(lapply(pairs,function(x)if(is.null(x$a_log))rep(NA_real_,length(x$a))else x$a_log),use.names=FALSE))
  b_log <- as.double(unlist(lapply(pairs,function(x)if(is.null(x$b_log))rep(NA_real_,length(x$b))else x$b_log),use.names=FALSE))
  supplied_valid <- function(p,lp) {
    valid <- is.finite(p)&is.finite(lp)&p>=0&p<=1&lp>=0
    nonzero <- valid&p>0
    valid[nonzero] <- abs(lp[nonzero]-stable_negative_log10(p[nonzero])) <= 1e-10+1e-7*abs(lp[nonzero])
    zero <- valid&p==0
    valid[zero] <- lp[zero] >= -log10(.Machine$double.xmin)-log10(.Machine$double.eps)
    valid
  }
  recovered <- !raw_comparable & supplied_valid(a,a_log)&supplied_valid(b,b_log)
  d <- rep(NA_real_,length(a)); d[comparable] <- abs(stable_negative_log10(a[comparable])-stable_negative_log10(b[comparable]))
  d[recovered] <- abs(a_log[recovered]-b_log[recovered])
  comparable <- comparable|recovered
  summary <- function(selection) {
    ok <- selection & comparable; values <- d[ok]; raw_ok <- selection & raw_comparable; raw <- abs(a[raw_ok]-b[raw_ok])
    list(total=sum(selection),comparable=sum(ok),recovered_with_native_log10=sum(selection&recovered),raw_uncomparable=sum(selection&!raw_comparable),uncomparable=sum(selection & !comparable),
      maximum_logp_absolute_difference=if(length(values)) max(values) else NULL,
      quantiles_logp_absolute_difference=if(length(values)) as.list(setNames(as.numeric(quantile(values,c(.5,.9,.95,.99,1),names=FALSE)),c("p50","p90","p95","p99","max"))) else NULL,
      maximum_raw_absolute_difference=if(length(raw))max(raw)else NULL,
      maximum_raw_relative_difference=if(length(raw))max(raw/a[raw_ok])else NULL,
      logp_failed_cells=sum(values>target),
      exceeded_0001=sum(values>1e-4),exceeded_001=sum(values>.001),exceeded_01=sum(values>.01),exceeded_05=sum(values>.05))
  }
  # Use positive raw reference P when available. Zero reference P enters
  # tail strata only when BOTH native logs passed the existing recovery gate.
  # This never changes comparable, d or the final numerical acceptance rule.
  recovered_reference_zero <- recovered & is.finite(a) & a==0
  # Positive raw P keeps the original comparisons exactly. Recovered zeros
  # additionally select their bins by the validated native -log10(P).
  recovered_below <- function(threshold) recovered_reference_zero & !is.na(a_log) & a_log>threshold
  # The unchanged recovery validity rule requires logP >= minimum-double
  # underflow scale (>323), so recovered zeros belong only in the first bin.
  strata <- list("(0,1e-100)"=(is.finite(a)&a>0&a<1e-100)|recovered_below(100),
    "[1e-100,1e-20)"=is.finite(a)&a>=1e-100&a<1e-20,
    "[1e-20,1e-8)"=is.finite(a)&a>=1e-20&a<1e-8,
    "[1e-8,1e-5)"=is.finite(a)&a>=1e-8&a<1e-5,
    "[1e-5,.05)"=is.finite(a)&a>=1e-5&a<.05,
    "[.05,1]"=is.finite(a)&a>=.05&a<=1)
  # Missing values, including equal NA, are recorded and cannot produce PASS.
  issue <- function(x) list(zero_underflow=sum(!is.na(x)&x==0),missing=sum(is.na(x)),
    nan=sum(is.nan(x)),infinite=sum(is.infinite(x)),
    out_of_range=sum(is.finite(x)&(x<0|x>1)))
  stored_a <- as.double(unlist(lapply(stored_pairs,function(x)x$a),use.names=FALSE))
  stored_b <- as.double(unlist(lapply(stored_pairs,function(x)x$b),use.names=FALSE))
  stored_ok <- is.finite(stored_a)&is.finite(stored_b)&stored_a>=0&stored_b>=0
  stored_difference <- abs(stored_a[stored_ok]-stored_b[stored_ok])
  supplied <- !is.na(a_log)|!is.na(b_log)
  stored_consistency_errors <- sum(supplied & (!supplied_valid(a,a_log)|!supplied_valid(b,b_log)))
  stored <- list(raw_probability_consistency_errors=stored_consistency_errors,total=length(stored_a),comparable=sum(stored_ok),
    uncomparable=sum(!stored_ok),definition="native -log10(P), positive for P<1; compare directly",
    maximum_absolute_difference=if(length(stored_difference))max(stored_difference)else NULL,
    quantiles_absolute_difference=if(length(stored_difference))as.list(setNames(as.numeric(quantile(stored_difference,c(.5,.9,.95,.99,1),names=FALSE)),c("p50","p90","p95","p99","max")))else NULL,
    exceeded_0001=sum(stored_difference>1e-4),exceeded_001=sum(stored_difference>1e-3),exceeded_01=sum(stored_difference>1e-2),
    failed_cells=sum(stored_difference>target))
  all_summary <- summary(rep(TRUE,length(a)))
  list(schema_version=1L,target_logp_absolute_difference=target,
    recognized_fields=recognized_fields,schema_errors=schema_errors,
    reference=issue(a),candidate=issue(b),all=all_summary,
    by_reference_p=lapply(strata,summary),
    small_reference_p=list(below_1e4=summary((is.finite(a)&a>0&a<1e-4)|recovered_below(4)),
      below_1e8=summary((is.finite(a)&a>0&a<1e-8)|recovered_below(8)),below_1e12=summary((is.finite(a)&a>0&a<1e-12)|recovered_below(12))),
    stored_pvalue_log10=stored,
    logp_passed=schema_errors==0L && length(a)>0 && all(comparable) && all(d<=target) && all(stored_ok) && all(stored_difference<=target) && stored_consistency_errors==0L,
    scope="Numerical acceptance is P-only: absolute -log10(P) error <= target; validate native structure and identifiers separately")
}
read_logp_native <- function(path) {
  if(tolower(tools::file_ext(path))=="rds") return(list(RDS=readRDS(path)))
  e <- new.env(parent=baseenv()); n <- load(path,e); mget(n,e,inherits=FALSE)
}
if(sys.nframe()==0L) {
  args <- commandArgs(TRUE)
  if(length(args)<3L) stop("Rscript validation/compare_logp.R expected.Rdata actual.Rdata report.json [logp_target]")
  suppressPackageStartupMessages(library(Matrix))
  suppressPackageStartupMessages(library(jsonlite))
  target <- if(length(args)>3L)as.double(args[4L])else .001
  report <- compare_logp_objects(read_logp_native(args[1L]),read_logp_native(args[2L]),target)
  write_json(report,args[3L],auto_unbox=TRUE,pretty=TRUE,digits=17,null="null")
  cat("recognized_p_cells:",report$all$total,"logp_comparable:",report$all$comparable,
      "logp_passed:",report$logp_passed,"\n")
  if(!report$logp_passed)quit(status=1L)
}
