# SPDX-License-Identifier: GPL-3.0-or-later
# Development oracle for https://github.com/li-lab-genetics/STAARpipeline.
# This script is never called by the Python production pipeline.
full_started <- proc.time()[[3]]
suppressPackageStartupMessages(library(STAARpipeline))
suppressPackageStartupMessages(library(SeqArray))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
config <- fromJSON(args[[1]],simplifyVector=FALSE)
manifest <- fromJSON(config$manifest,simplifyVector=FALSE)
parts <- strsplit(args[[2]],':',fixed=TRUE)[[1]]
kind <- parts[[1]];group <- as.integer(parts[[2]])
stopifnot(kind %in% c('coding','noncoding','ncrna','individual'),group>=1L)
out <- config$output_directory;dir.create(out,recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out,'entries'),showWarnings=FALSE)
model <- readRDS(config$null_model)
stopifnot(model$n.pheno==1L,!model$use_SPA,model$relatedness,model$sparse_kins)
gds <- seqOpen(config$gds)
all_ids <- seqGetData(gds,'sample.id')
stopifnot(all(model$id_include %in% all_ids))
seqSetFilter(gds,sample.id=model$id_include)
selected_ids <- as.character(seqGetData(gds,'sample.id'))
stopifnot(length(selected_ids)==length(model$id_include),!anyDuplicated(selected_ids),setequal(selected_ids,as.character(model$id_include)))
native_order_matches_model <- identical(selected_ids,as.character(model$id_include))
seqResetFilter(gds)
catalog_json <- fromJSON(config$annotation_catalog,simplifyVector=TRUE)
catalog <- data.frame(name=names(catalog_json),dir=unlist(catalog_json,use.names=FALSE),stringsAsFactors=FALSE)
annotation_names <- unlist(manifest$annotation_names,use.names=FALSE)
stopifnot(all(annotation_names %in% catalog$name))
common <- list(chr=as.integer(manifest$chromosome),genofile=gds,obj_nullmodel=model,
 QC_label='annotation/info/QC_label',variant_type='SNV',geno_missing_imputation='mean',
 rare_maf_cutoff=0.01,rv_num_cutoff=2,rv_num_cutoff_max=1e9,rv_num_cutoff_max_prefilter=1e9,
 Annotation_dir='',Annotation_name_catalog=catalog,Use_annotation_weights=TRUE,
 Annotation_name=annotation_names,silent=FALSE)
array_id <- as.integer(manifest$array_offsets[[kind]])+group
prefix <- if(is.null(config$output_prefix))'Phenotype'else config$output_prefix
entry_times <- list();coverage <- list()
record_coverage <- function(result,gene_name,index) {
 masks <- if(kind %in% c('coding','noncoding'))result else list(ncRNA=result)
 lapply(names(masks),function(mask){x<-masks[[mask]];list(gene_name=gene_name,catalog_index=index,mask=mask,
  output_empty=is.null(x)||length(x)==0L,typeof=typeof(x),class=class(x),dimensions=dim(x),column_names=colnames(x))})
}
if(kind=='individual') {
 start <- as.double(manifest$start_loc)+(group-1)*as.double(manifest$individual_region_size)
 end <- min(start+as.double(manifest$individual_region_size)-1,as.double(manifest$end_loc))
 stopifnot(start<=end)
 call_args <- list(chr=as.integer(manifest$chromosome),start_loc=start,end_loc=end,genofile=gds,obj_nullmodel=model,
  mac_cutoff=20,QC_label='annotation/info/QC_label',variant_type='variant',geno_missing_imputation='mean',silent=FALSE)
 call_args <- call_args[names(call_args)%in%names(formals(Individual_Analysis))]
 started <- proc.time()[[3]]
 results_individual_analysis <- do.call(Individual_Analysis,call_args)
 entry_times[[1]] <- list(start_loc=start,end_loc=end,seconds=proc.time()[[3]]-started,
                         returned_rows=if(is.null(results_individual_analysis))0L else nrow(results_individual_analysis))
 filename <- sprintf('%s_Individual_Analysis_%d.Rdata',prefix,array_id)
 save(results_individual_analysis,file=file.path(out,filename))
} else {
 genes <- if(kind=='ncrna')manifest$ncRNA_genes else manifest$genes_info
 per_batch <- as.integer(if(kind=='ncrna')manifest$ncrna_genes_per_batch else manifest$coding_genes_per_batch)
 first <- (group-1L)*per_batch+1L;last <- min(group*per_batch,length(genes))
 stopifnot(first<=last)
 combined <- c()
 for(index in first:last) {
  gene <- genes[[index]];call_args <- common;call_args$gene_name <- gene$gene_name
  fn <- if(kind=='coding')Gene_Centric_Coding else if(kind=='noncoding')Gene_Centric_Noncoding else ncRNA
  if(kind=='coding')call_args$category<-'all_categories_incl_ptv'
  if(kind=='noncoding')call_args$category<-'all_categories'
  seqResetFilter(gds)
  started <- proc.time()[[3]]
  result <- do.call(fn,call_args)
  entry_times[[length(entry_times)+1L]] <- list(gene_name=gene$gene_name,catalog_index=gene$catalog_index,
                                             seconds=proc.time()[[3]]-started)
  coverage <- c(coverage,record_coverage(result,gene$gene_name,gene$catalog_index))
  saveRDS(result,file.path(out,'entries',sprintf('%s_%05d.rds',kind,index)))
  combined <- if(kind=='ncrna')rbind(combined,result)else append(combined,result)
  cat('ENTRY_COMPLETE',kind,index,gene$gene_name,'\n');flush.console()
 }
 if(kind=='coding') {results_coding<-combined;filename<-sprintf('%s_Coding_%d.Rdata',prefix,array_id);save(results_coding,file=file.path(out,filename))}
 if(kind=='noncoding') {results_noncoding<-combined;filename<-sprintf('%s_Noncoding_%d.Rdata',prefix,array_id);save(results_noncoding,file=file.path(out,filename))}
 if(kind=='ncrna') {results_ncRNA<-combined;filename<-sprintf('%s_ncRNA_%d.Rdata',prefix,array_id);save(results_ncRNA,file=file.path(out,filename))}
}
seqClose(gds)
report <- list(kind=kind,group=group,array_id=array_id,filename=filename,number_samples=length(model$id_include),
 native_order_matches_model=native_order_matches_model,
 package_versions=list(STAAR=as.character(packageVersion('STAAR')),STAARpipeline=as.character(packageVersion('STAARpipeline'))),
 script_wall_seconds=proc.time()[[3]]-full_started,function_seconds=sum(vapply(entry_times,function(x)x$seconds,numeric(1))),
 entry_times=entry_times,coverage=coverage,blas_environment=as.list(Sys.getenv(c('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))))
write_json(report,file.path(out,sprintf('timing_%s_%d.json',kind,group)),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
cat('BATCH_COMPLETE',kind,group,array_id,'script_seconds',report$script_wall_seconds,'\n')
