## IQ-Tree Script for ASR of PF0065 ## 

# convert data from .txt to .fasta

library(phylotools)
library(tidyverse)

# read in data
setwd("~/Desktop/asr-ae/data/Ding/processed")
pf_00041 <- read.table("./PF00041/seq_msa_char_pf00041.txt")
pf_00067 <- read.table("./PF00067/seq_msa_char_pf00067.txt")
pf_00565 <- read.table("./PF00565/seq_msa_char_pf00565.txt")
pf_00144 <- read.table("./PF00144/seq_msa_char_pf00144.txt")

# change column names to be compatible with dat2fasta
colnames(pf_00041) <- c("seq.name", "seq.text")
colnames(pf_00067) <- c("seq.name", "seq.text")
colnames(pf_00565) <- c("seq.name", "seq.text")
colnames(pf_00144) <- c("seq.name", "seq.text")

# write to FASTA
dat2fasta(pf_00041, outfile = "~/Desktop/asr-ae/data/Ding/processed/PF00041/seq_msa_char_pf00041.fasta")
dat2fasta(pf_00067, outfile = "~/Desktop/asr-ae/data/Ding/processed/PF00067/seq_msa_char_pf00067.fasta")
dat2fasta(pf_00565, outfile = "~/Desktop/asr-ae/data/Ding/processed/PF00565/seq_msa_char_pf00565.fasta")
dat2fasta(pf_00144, outfile = "~/Desktop/asr-ae/data/Ding/processed/PF00144/seq_msa_char_pf00144.fasta")

# connect to WID server 

`ssh hlouw@solislemus-001.discover.wisc.edu`

# IQ-Tree folder 

`cd hlouw@solislemus-001.discovery.wisc.edu/mnt/ws/home/hlouw/iqtree-project`

# IQ-Tree command with 1000 Ultrafast Bootstrap replicates

Session 0: `./iqtree-1.6.12-Linux/bin/iqtree -s /mnt/ws/home/hlouw/iqtree-project/seq_msa_char_pf00041.fasta -m MFP -bb 1000`
Session 1: `./iqtree-1.6.12-Linux/bin/iqtree -s /mnt/ws/home/hlouw/iqtree-project/seq_msa_char_pf00067.fasta -m MFP -bb 1000`
Session 2: `./iqtree-1.6.12-Linux/bin/iqtree -s /mnt/ws/home/hlouw/iqtree-project/seq_msa_char_pf00565.fasta -m MFP -bb 1000`
Session 3: `./iqtree-1.6.12-Linux/bin/iqtree -s /mnt/ws/home/hlouw/iqtree-project/seq_msa_char_pf00144.fasta -m MFP -bb 1000`


