## Align training data with MUSCLE for DNA encoder

## Training data generation script found in `./Desktop/asr-ae/scripts/asr-ae-dat.R`

Raw data: `./Desktop/asr-ae/data/train_dat.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./train_dat.fasta -out ./train_dat_aligned_MUCSLE.fasta`


## CARD1 for training
Raw data: `./Desktop/asr-ae/data/raxml-ng/data/structaligned_card1.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./raxml-ng/data/structaligned_card1.fasta -out ./raxml-ng/data/card1_aligned_MUCSLE.fasta`

## RD1 for training
Raw data: `./Desktop/asr-ae/data/raxml-ng/data/structaligned_rd1.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./raxml-ng/data/structaligned_rd1.fasta -out ./raxml-ng/data/rd1_aligned_MUCSLE.fasta`

## DRSM1 for training
Raw data: `./Desktop/asr-ae/data/raxml-ng/data/structaligned_drsm1.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./raxml-ng/data/structaligned_drsm1.fasta -out ./raxml-ng/data/drsm1_aligned_MUCSLE.fasta`


## DRSM 3 for testing

Raw data: `./Desktop/asr-ae/data/raxml-ng/data/structaligned_drsm3.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./raxml-ng/data/structaligned_drsm3.fasta -out ./raxml-ng/data/drsm3_aligned_MUCSLE.fasta`

## DRSM 2 for downstream testing

Raw data: `./Desktop/asr-ae/data/raxml-ng/data/structaligned_drsm2.fasta`

Directory: `cd ./Desktop/asr-ae/data`

Line: `./muscle -in ./raxml-ng/data/structaligned_drsm2.fasta -out ./raxml-ng/data/drsm2_aligned_MUCSLE.fasta`

