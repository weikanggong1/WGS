# SPDX-License-Identifier: GPL-3.0-only
# Development-only original-package gene catalog export; no genetic data are redistributed.
# Args: chromosome output_dir [genes_per_array] [rna_per_array] [jobs_num.Rdata] [input.gds]
suppressPackageStartupMessages(library(STAARpipeline))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2)
chr <- as.integer(args[[1]]);out <- args[[2]]
gene_per_array <- if(length(args)>=3)as.integer(args[[3]])else 50L
rna_per_array <- if(length(args)>=4)as.integer(args[[4]])else 100L
dir.create(out,recursive=TRUE,showWarnings=FALSE)
ns <- asNamespace('STAARpipeline')
genes_raw <- get('genes_info',envir=ns)
rna_raw <- get('ncRNA_gene',envir=ns)
gene_all <- as.data.frame(genes_raw,stringsAsFactors=FALSE)
rna_all <- as.data.frame(rna_raw,stringsAsFactors=FALSE)
stopifnot(ncol(gene_all)>=4L,ncol(rna_all)>=2L)
gene_rows <- which(as.integer(as.character(gene_all[[2]]))==chr)
rna_rows <- which(as.integer(as.character(rna_all[[1]]))==chr)
manifest <- data.frame(catalog_index=gene_rows,chromosome=as.character(gene_all[[2]][gene_rows]),
 gene_name=as.character(gene_all[[1]][gene_rows]),start=as.integer(as.character(gene_all[[3]][gene_rows])),
 end=as.integer(as.character(gene_all[[4]][gene_rows])),stringsAsFactors=FALSE)
rna_manifest <- data.frame(catalog_index=rna_rows,chromosome=as.character(rna_all[[1]][rna_rows]),
 gene_name=as.character(rna_all[[2]][rna_rows]),stringsAsFactors=FALSE)
write.table(manifest,file.path(out,'genes_info.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(rna_manifest,file.path(out,'ncRNA_gene.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(gene_all,file.path(out,'genes_info_full_raw.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(rna_all,file.path(out,'ncRNA_gene_full_raw.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
make_batches <- function(raw_chr,manifest,per_array,kind) {
 counts <- table(raw_chr);groups <- ceiling(counts/per_array)
 # Reproduce the tutorial's numeric chromosome→array offset expression.
 previous <- if(chr<=1L)0L else sum(groups[seq_len(chr-1L)])
 n <- nrow(manifest);pieces <- split(seq_len(n),ceiling(seq_len(n)/per_array))
 lapply(seq_along(pieces),function(group)list(kind=kind,chromosome=chr,arrayid=as.integer(previous+group),
  groupid=group,catalog_indices=as.integer(manifest$catalog_index[pieces[[group]]]),
  row_start=min(pieces[[group]]),row_end=max(pieces[[group]]),gene_names=manifest$gene_name[pieces[[group]]]))
}
metadata <- list(package_version=as.character(packageVersion('STAARpipeline')),chromosome=chr,
 source_objects=list(genes_info=list(class=class(genes_raw),typeof=typeof(genes_raw),dim=dim(genes_raw),colnames=colnames(genes_raw)),
 ncRNA_gene=list(class=class(rna_raw),typeof=typeof(rna_raw),dim=dim(rna_raw),colnames=colnames(rna_raw))),
 chromosome_counts=list(genes_info=as.list(table(gene_all[[2]])),ncRNA_gene=as.list(table(rna_all[[1]]))),
 genes=nrow(manifest),ncRNA_genes=nrow(rna_manifest),include_empty_masks=TRUE,
 coding=make_batches(gene_all[[2]],manifest,gene_per_array,'coding'),
 noncoding=make_batches(gene_all[[2]],manifest,gene_per_array,'noncoding'),
 ncrna=make_batches(rna_all[[1]],rna_manifest,rna_per_array,'ncrna'))
write_json(metadata,file.path(out,'manifest.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
if(length(args)>=5L) {
  e <- new.env();loaded <- load(args[[5]],envir=e)
  jobs_num <- get(loaded[[1]],envir=e)
  stopifnot(all(c('start_loc','end_loc','individual_analysis_num') %in% names(jobs_num)))
  bounds <- c(as.double(jobs_num$start_loc[chr]),as.double(jobs_num$end_loc[chr]))
  if(length(args)>=6L) {
    suppressPackageStartupMessages(library(SeqArray))
    gds <- seqOpen(args[[6]])
    position <- seqGetData(gds,'position')
    qc <- as.character(seqGetData(gds,'annotation/info/QC_label'))
    actual_bounds <- range(position[which(qc=='PASS')])
    seqClose(gds)
    stopifnot(identical(as.double(actual_bounds),bounds))
  }
  gene_groups <- ceiling(table(gene_all[[2]])/gene_per_array)
  rna_groups <- ceiling(table(rna_all[[1]])/rna_per_array)
  previous <- function(x)if(chr<=1L)0L else as.integer(sum(x[seq_len(chr-1L)]))
  root_manifest <- list(chromosome=chr,
    genes_info=lapply(seq_len(nrow(manifest)),function(i)list(gene_name=manifest$gene_name[[i]],start=manifest$start[[i]],end=manifest$end[[i]],catalog_index=manifest$catalog_index[[i]])),
    ncRNA_genes=lapply(seq_len(nrow(rna_manifest)),function(i)list(gene_name=rna_manifest$gene_name[[i]],catalog_index=rna_manifest$catalog_index[[i]])),
    start_loc=bounds[[1]],end_loc=bounds[[2]],
    array_offsets=list(coding=previous(gene_groups),noncoding=previous(gene_groups),ncrna=previous(rna_groups),individual=previous(jobs_num$individual_analysis_num)),
    coding_genes_per_batch=gene_per_array,ncrna_genes_per_batch=rna_per_array,individual_region_size=10000000L,
    number_coding_genes=nrow(manifest),number_ncRNA_genes=nrow(rna_manifest),number_individual_arrays=as.integer(jobs_num$individual_analysis_num[chr]),
    pass_extrema_checked=length(args)>=6L,package_version=as.character(packageVersion('STAARpipeline')),
    annotation_names=c('CADD','LINSIGHT','FATHMM.XF','aPC.EpigeneticActive','aPC.EpigeneticRepressed','aPC.EpigeneticTranscription','aPC.Conservation','aPC.LocalDiversity','aPC.Mappability','aPC.TF','aPC.Protein'))
  write_json(root_manifest,file.path(out,'root_manifest.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
  cat('ROOT_MANIFEST',bounds,'array_offsets',unlist(root_manifest$array_offsets),'\n')
}

cat('MANIFEST chr',chr,'coding/noncoding',nrow(manifest),'ncRNA',nrow(rna_manifest),'\n')
print(metadata$source_objects)
