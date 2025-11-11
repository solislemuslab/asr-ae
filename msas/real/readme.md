For PFAM protein family PF00565 (Staphylococcal nuclease homologues), which was also analyzed by [Ding](https://www.nature.com/articles/s41467-019-13633-0), we downloaded the raw "full" MSA (v. 21) to `msas/real/raw/PF00565.stk`

For PFAM protein family PF00144 (beta-lactamase), which was analyzed by [Detlefsen](https://www.nature.com/articles/s41467-022-29443-w), we download `https://github.com/MachineLearningLifeScience/meaningful-protein-representations/blob/master/tape/PF00144_full.txt`, which we assume was obtained from the full alignment of the family from PFAM, to `msas/real/raw/PF00144.fasta`

To generate the files in the directory `msas/real/processed/PF00565`, we run 
``` 
python scripts/process_msa.py msas/real/raw/PF00565.stk --real --query SND1_HUMAN/552-660
```
Note that specifically for this family (PF00565), the pre-processing script filters out any sequence that is not from a Eukaryote, i.e. any sequence that's not listed in `msas/real/PF00565_eukaryotes.tsv`

To generate the files in the directory `msas/real/processed/PF00144_og`, we had to run a previous version of the pre-processing script with query sequence "A0A010Q9K6_9PEZI/15-292"
