# Format controls; not a scientific benchmark.
source('validation/compare_structure.R')
check_pass <- function(a,b) {
  report <- compare_structure_objects(a,b)
  if(!report$passed) {print(report);str(a);str(b)}
  stopifnot(report$passed)
}
a <- list(result=list(data.frame(id=c('a','b'), kind=factor(c('SNV','Indel')),
                                pvalue=c(1e-30,.5), Score=c(1,2)), NULL))
b <- a; b$result[[1]]$Score <- c(4,8); b$result[[1]]$pvalue <- c(1e-25,.6)
# R 3.6 column replacement rebuilds attribute order; preserve the schema.
attributes(b$result[[1]]) <- attributes(a$result[[1]])
check_pass(a,b)
b$result[[1]]$id <- rev(b$result[[1]]$id)
stopifnot(!compare_structure_objects(a,b)$passed)
b <- a; b$result[[2]] <- numeric(0)
stopifnot(!compare_structure_objects(a,b)$passed)
b <- a; b$result[[1]]$Score <- as.integer(b$result[[1]]$Score)
stopifnot(!compare_structure_objects(a,b)$passed)
b <- a; b$result[[1]]$Score[1] <- NA_real_
stopifnot(!compare_structure_objects(a,b)$passed)
b <- a; levels(b$result[[1]]$kind) <- c('changed','SNV')
stopifnot(!compare_structure_objects(a,b)$passed)
a <- matrix(list('id',1.5,2L,NULL), nrow=1L,
            dimnames=list(NULL,c('label','float','index','empty')))
b <- a; b[[2]] <- 3.0
check_pass(a,b)
b[[3]] <- 3L
stopifnot(!compare_structure_objects(a,b)$passed)
if (requireNamespace('Matrix',quietly=TRUE)) {
  a <- Matrix::sparseMatrix(i=c(1,2),j=c(1,2),x=c(1,2))
  b <- a; b@x <- c(2,3)
  check_pass(a,b)
  b <- a; b@i[1] <- 1L
  stopifnot(!compare_structure_objects(a,b)$passed)
}
cat('NATIVE_STRUCTURE_CONTROLS_PASS\n')
