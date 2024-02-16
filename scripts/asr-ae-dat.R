library(seqinr)
library(Biostrings)
library(phylotools)
library(dplyr)
library(tidyr)
library(stringr)
library(EnvNJ)

setwd("~/Desktop/asr-ae/data/")

# Read in files
card1 <- read.fasta("./raxml-ng/data/structaligned_card1.fasta", seqtype = "AA")
drsm1 <- read.fasta("./raxml-ng/data/structaligned_drsm1.fasta", seqtype = "AA")
rd1 <- read.fasta("./raxml-ng/data/structaligned_rd1.fasta", seqtype = "AA")

# Concatenate into one file
train_dat <- c(card1, drsm1, rd1)

# try with different package
card1 <- readDNAStringSet("~/Desktop/asr-ae/data/raxml-ng/data/structaligned_card1.fasta")
seq_name <- names(card1)
sequence <- paste(card1)
card1 <- data.frame(seq_name, sequence)

drsm1 <- readDNAStringSet("~/Desktop/asr-ae/data/raxml-ng/data/structaligned_drsm1.fasta")
seq_name <- names(drsm1)
sequence <- paste(drsm1)
drsm1 <- data.frame(seq_name, sequence)

rd1 <- readDNAStringSet("~/Desktop/asr-ae/data/raxml-ng/data/structaligned_rd1.fasta")
seq_name <- names(rd1)
sequence <- paste(rd1)
rd1 <- data.frame(seq_name, sequence)

train_dat <- rbind(card1, drsm1, rd1)
colnames(train_dat) <- c("seq.name", "seq.text")

dat2fasta(train_dat, outfile = "~/Desktop/asr-ae/data/train_dat.fasta")


#writeFasta<-function(data, filename){
  fastaLines = c()
  for (rowNum in 1:nrow(data)){
    fastaLines = c(fastaLines, as.character(paste(">", data[rowNum,"name"], sep = "")))
    fastaLines = c(fastaLines,as.character(data[rowNum,"seq"]))
  }
  fileConn<-file(filename)
  writeLines(fastaLines, fileConn)
  close(fileConn)
}

#writeFasta(train_dat, "~/Desktop/asr-ae/data/asr-ae-train.fasta")




