## PF00072
The PFAM protein family PF00072 saved as `raw/PF00072.fa` is the alignment saved as `aln/1jbeA.aln` in `http://bioinf.cs.ucl.ac.uk/downloads/PSICOV/suppdata/`, which was published with the [PSICOV paper](https://academic.oup.com/bioinformatics/article/28/2/184/198108)

To generate files in the directory `msas/real/processed/PF00072`, we run 
``` 
python scripts/process_msa.py msas/real/raw/PF00072.fa --real --query Seq190
```
Seq190 is the sequence used as the "target" in the PSICOV paper and thus, includes no gaps in the raw alignment.

## PF00565
The PFAM protein family PF00565 (Staphylococcal nuclease homologues) saved as `raw/PF00565.stk`, is the "full" MSA  downloaded from PFAM (v. 21).

To generate the files in the directory `msas/real/processed/PF00565`, we run 
``` 
python scripts/process_msa.py msas/real/raw/PF00565.stk --real --query SND1_HUMAN/552-660
```
Note that specifically for this family (PF00565), the pre-processing script filters out any sequence that is not from a Eukaryote, i.e. any sequence that's not listed in `msas/real/PF00565_eukaryotes.tsv`

## PF00144
The PFAM protein family PF00144 (beta-lactamase) saved as `raw/PF00144.fa` is taken from `https://github.com/MachineLearningLifeScience/meaningful-protein-representations/blob/master/tape/PF00144_full.txt`.

To generate files in the directory `msas/real/processed/PF00144_og`, we ran a previous version of the pre-processing script with query sequence "A0A010Q9K6_9PEZI/15-292"





