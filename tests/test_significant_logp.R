# CPU-only validator unit tests; synthetic structures test parser contracts,
# never substitute for a scientific benchmark.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)%in%c(2L,3L))
source(args[[1L]])
compare <- significant_logp_function(args[[2L]])
reports <- list()
check <- function(a,b,passed,total,failed=0L) {
  r <- compare(a,b,.001)
  stopifnot(identical(r$significant_logp_passed,passed),r$significant_logp$total==total,
            r$significant_logp$logp_failed_cells==failed)
  reports[[length(reports)+1L]] <<- r
  r
}
tests <- 0L
r <- check(list(pvalue=.6),list(pvalue=.9),TRUE,0);stopifnot(!r$logp_passed,r$all$total==1L,r$all$comparable==1L);tests<-tests+1L
r <- check(list(pvalue=.04999),list(pvalue=.05001),TRUE,1);stopifnot(r$significant_logp$boundary_crossing_cells==1L,r$significant_logp$reference_only_cells==1L);tests<-tests+1L
r <- check(list(pvalue=.05001),list(pvalue=.04999),TRUE,1);stopifnot(r$significant_logp$candidate_only_cells==1L);tests<-tests+1L
r <- check(list(pvalue=.04999),list(pvalue=.051),FALSE,1,1);tests<-tests+1L
r <- check(list(pvalue=.05),list(pvalue=.049999),TRUE,1);stopifnot(r$significant_logp$reference_equal_threshold_cells==1L);tests<-tests+1L
r <- check(list(pvalue=.05),list(pvalue=.06),TRUE,0);tests<-tests+1L
r <- check(list(pvalue=NA_real_),list(pvalue=NA_real_),FALSE,0);stopifnot(r$all$uncomparable==1L);tests<-tests+1L
r <- check(list(pvalue=NaN),list(pvalue=NaN),FALSE,0);tests<-tests+1L
r <- check(list(pvalue=Inf),list(pvalue=Inf),FALSE,0);tests<-tests+1L
r <- check(list(pvalue=-.1),list(pvalue=-.1),FALSE,0);tests<-tests+1L
r <- check(list(pvalue=1.1),list(pvalue=1.1),FALSE,0);tests<-tests+1L
r <- check(list(pvalue=0),list(pvalue=0),FALSE,1);stopifnot(r$all$uncomparable==1L);tests<-tests+1L
r <- check(list(pvalue=1),list(pvalue=1),TRUE,0);tests<-tests+1L
r <- check(data.frame(pvalue=0,pvalue_log10=400),data.frame(pvalue=0,pvalue_log10=400.0005),TRUE,1);stopifnot(r$all$recovered_with_native_log10==1L,r$reference$zero_underflow==1L);tests<-tests+1L
r <- check(data.frame(pvalue=0,pvalue_log10=300),data.frame(pvalue=0,pvalue_log10=300),FALSE,1);tests<-tests+1L
r <- check(data.frame(pvalue=.6,pvalue_log10=1),data.frame(pvalue=.6,pvalue_log10=1),FALSE,0);stopifnot(r$stored_pvalue_log10$raw_probability_consistency_errors>0L);tests<-tests+1L
r <- check(list(pvalue=1L),list(pvalue=1L),FALSE,0);stopifnot(r$schema_errors>0L);tests<-tests+1L
r <- check(list(nested=list(c(`SKAT(1,25)-annotation`=.04,`STAAR-O`=.8))),list(nested=list(c(`SKAT(1,25)-annotation`=.04001,`STAAR-O`=.9))),TRUE,1);stopifnot(r$all$total==2L);tests<-tests+1L
r <- check(list(),list(),FALSE,0);stopifnot(!r$logp_passed,r$all$total==0L);tests<-tests+1L
cat("significant_validator_CPU_tests_passed:",tests,"\n")
if(length(args)==3L)jsonlite::write_json(reports,args[[3L]],auto_unbox=TRUE,pretty=TRUE,digits=17,null="null")
