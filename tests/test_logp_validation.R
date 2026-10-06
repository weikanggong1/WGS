# Meaningful schema/underflow controls; no scientific benchmark is synthetic.
source('validation/compare_logp.R')
a <- list(data.frame(pvalue=c(1e-30,.5,1),MAF=c(.1,.2,.3)))
b <- a; b[[1]]$MAF[] <- c(.9,.8,.7); attributes(b[[1]]) <- attributes(a[[1]])
x <- compare_logp_objects(a,b)
stopifnot(x$all$total==3L,x$logp_passed,x$all$maximum_logp_absolute_difference==0)
b[[1]]$pvalue[1] <- 1e-20
x <- compare_logp_objects(a,b);stopifnot(!x$logp_passed,x$all$maximum_logp_absolute_difference==10)
a <- list(data.frame(pvalue=c(0,NA_real_,Inf,-.1,1.1)))
x <- compare_logp_objects(a,a)
stopifnot(!x$logp_passed,x$all$uncomparable==5,x$reference$zero_underflow==1,x$reference$out_of_range==2)
a <- list(matrix(list(1e-100,.2,'label',.3),2,dimnames=list(NULL,c('STAAR-O','metadata'))))
x <- compare_logp_objects(a,a);stopifnot(x$logp_passed,x$all$total==2)
stopifnot(!any(logp_field(c('BETA','MAF','CHISQ','SKAT-statistic','pvalue_log10'))))
stopifnot(all(logp_field(c('SKAT(1,25)-CADD','Burden(1,1)','STAAR-S(1,25)','ACAT-O'))))
stopifnot(identical(stable_negative_log10(1),0),stable_negative_log10(1-1e-15)>0)
b <- a;colnames(b[[1]]) <- c('MAF','metadata')
stopifnot(!compare_logp_objects(a,b)$logp_passed)
cat('LOGP_CONTROLS_PASS\n')
a <- list(data.frame(pvalue=c(0,1e-100),pvalue_log10=c(400,100)))
b <- a;b[[1]]$pvalue_log10[1] <- 400.0005
attributes(b[[1]]) <- attributes(a[[1]])
x <- compare_logp_objects(a,b)
stopifnot(x$logp_passed,x$all$recovered_with_native_log10==1,x$all$raw_uncomparable==1)
b[[1]]$pvalue_log10[1] <- 200
stopifnot(!compare_logp_objects(a,b)$logp_passed)
a <- list(data.frame(pvalue=.1,pvalue_log10=-1))
stopifnot(!compare_logp_objects(a,a)$logp_passed)
cat('LOGP_NATIVE_LOG_CONTROLS_PASS\n')
a <- list(data.frame(pvalue=c(0,.1),pvalue_log=c(400,1)*log(10)))
stopifnot(compare_logp_objects(a,a)$logp_passed)
b <- a;b[[1]]$pvalue_log[2] <- 2*log(10)
stopifnot(!compare_logp_objects(a,b)$logp_passed)
cat('LOGP_NATIVE_LN_CONTROLS_PASS\n')

a <- list(matrix(list(NULL,NULL),1,dimnames=list(NULL,c("mask_a","mask_b"))))
x <- compare_logp_objects(a,a)
stopifnot(x$recognized_fields==0L,x$schema_errors==0L,x$all$total==0L,!x$logp_passed)
b <- a;colnames(b[[1]]) <- c("mask_a","changed_mask")
stopifnot(compare_logp_objects(a,b)$schema_errors>0L)
cat("LOGP_EMPTY_NATIVE_CONTROLS_PASS\n")

# Named atomic fields are recognized even when their scalar differences are
# below the old absolute raw-P tolerance.
a <- list(c(pvalue=1e-40,MAF=.2)); b <- a;b[[1]]['pvalue'] <- 1e-20
x <- compare_logp_objects(a,b)
stopifnot(x$recognized_fields==1L,x$all$total==1L,!x$logp_passed,
          abs(x$all$maximum_logp_absolute_difference-20)<1e-12)
a <- list(c('SKAT(1,25)'=1e-30,'Burden(1,1)'=.2,'STAAR-O'=.1,Score=2))
stopifnot(compare_logp_objects(a,a)$all$total==3L,compare_logp_objects(a,a)$logp_passed)
b <- a;b[[1]]['Score'] <- 20
stopifnot(compare_logp_objects(a,b)$logp_passed) # non-P contract remains a separate checker

# Atomic log fields pair with raw pvalue, validate sign/units/consistency, and
# participate in the independent native-log comparison.
a <- list(c(pvalue=0,pvalue_log10=400,MAF=.2));b <- a
b[[1]]['pvalue_log10'] <- 400.0005
x <- compare_logp_objects(a,b)
stopifnot(x$logp_passed,x$all$recovered_with_native_log10==1L,
          x$stored_pvalue_log10$total==1L,x$stored_pvalue_log10$failed_cells==0L)
for(field in c('(0,1e-100)')) stopifnot(x$by_reference_p[[field]]$total==1L,
  x$by_reference_p[[field]]$recovered_with_native_log10==1L)
stopifnot(x$small_reference_p$below_1e4$total==1L,
          x$small_reference_p$below_1e8$total==1L,
          x$small_reference_p$below_1e12$total==1L)
b[[1]]['pvalue_log10'] <- 400.01;stopifnot(!compare_logp_objects(a,b)$logp_passed)
a <- list(c(pvalue=1e-30,pvalue_log=30*log(10)))
stopifnot(compare_logp_objects(a,a)$logp_passed,compare_logp_objects(a,a)$stored_pvalue_log10$total==1L)
b <- a;b[[1]]['pvalue_log'] <- -30*log(10);stopifnot(!compare_logp_objects(a,b)$logp_passed)
b <- a;b[[1]]['pvalue_log'] <- 31*log(10);stopifnot(!compare_logp_objects(a,b)$logp_passed)

# Existing data.frame recovery is additionally counted in small-P bins; gate
# values and native-log errors remain those of the original implementation.
a <- list(data.frame(pvalue=c(0,1e-100,.05,1),pvalue_log10=c(400,100,-log10(.05),0)))
x <- compare_logp_objects(a,a)
stopifnot(x$logp_passed,x$all$total==4L,x$all$comparable==4L,
          x$all$recovered_with_native_log10==1L,
          x$all$maximum_logp_absolute_difference==0,
          x$by_reference_p[['(0,1e-100)']]$total==1L,
          x$by_reference_p[['[1e-100,1e-20)']]$total==1L,
          x$by_reference_p[['[.05,1]']]$total==2L)
# Bare zero, implausible native zero recovery, and missing values remain failures.
for(a in list(list(c(pvalue=0)),list(c(pvalue=0,pvalue_log10=200)),
              list(c(pvalue=NA_real_,pvalue_log10=400)),
              list(data.frame(pvalue=0,pvalue_log10=NA_real_)))) {
    invalid <- compare_logp_objects(a,a)
    stopifnot(!invalid$logp_passed,invalid$all$recovered_with_native_log10==0L,
              invalid$by_reference_p[['(0,1e-100)']]$total==0L)
}
# Inherited P-vector semantics and schema/type/attribute checks stay unchanged.
a <- list(pvalue=setNames(c(.1,.2),c('observation_a','observation_b')))
inherited <- compare_logp_objects(a,a)
stopifnot(inherited$logp_passed,inherited$recognized_fields==1L,inherited$all$total==2L)
a <- list(c(pvalue=.1,pvalue_log10=1)); b <- a
names(b[[1]]) <- rev(names(b[[1]]));stopifnot(compare_logp_objects(a,b)$schema_errors>0L)
b <- a;attr(b[[1]],'unit') <- 'changed';stopifnot(compare_logp_objects(a,b)$schema_errors>0L)
b <- list(as.integer(a[[1]]));stopifnot(compare_logp_objects(a,b)$schema_errors>0L)
# All original positive-reference bin endpoints and complete reports unchanged.
a <- list(data.frame(pvalue=c(1e-101,1e-100,1e-20,1e-8,1e-5,.05,1),
                     pvalue_log10=-log10(c(1e-101,1e-100,1e-20,1e-8,1e-5,.05,1))))
endpoint <- compare_logp_objects(a,a)
stopifnot(endpoint$logp_passed,endpoint$all$total==7L,
          identical(as.integer(vapply(endpoint$by_reference_p,function(bin)bin$total,numeric(1))),c(1L,1L,1L,1L,1L,2L)),
          endpoint$small_reference_p$below_1e4$total==5L,
          endpoint$small_reference_p$below_1e8$total==3L,
          endpoint$small_reference_p$below_1e12$total==3L)
cat('LOGP_NAMED_VECTOR_AND_RECOVERED_STRATA_PASS\n')
