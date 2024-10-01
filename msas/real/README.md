For PFAM protein family 00565 (Staphylococcal nuclease homologues), which was analyzed by [Ding](https://www.nature.com/articles/s41467-019-13633-0), we downloaded the raw "full" MSA from the Interpro website.

For PFAM protein family PF00144 (beta-lactamase), which was analyzed by [Detlefsen](https://www.nature.com/articles/s41467-022-29443-w), we download `https://github.com/MachineLearningLifeScience/meaningful-protein-representations/blob/master/tape/PF00144_full.txt`, which we assume was obtained from the full alignment of the family from Interpro.

Finally, for PF02033 (Ribosome-binding factor A), we download an alignment the alignment found at `aln/1josA.aln` of `http://bioinf.cs.ucl.ac.uk/downloads/PSICOV/suppdata/`, which was published with the [PSICOV paper](https://academic.oup.com/bioinformatics/article/28/2/184/198108). We don't have meaningful sequence labels for this family.

To each of these MSAs, we then applied the pre-processing steps from the Ding paper. Specifically for PF00565, this entails first filtering the alignment to retain only sequences from Eukaryotic species + 1 sequence from a non-Eukaryotic species as an outgroup for tree rooting. For all families, however, it entails specifying a query sequence, which is recorded below, and filtering to retain only sequences that do not have too many gaps in the positions where the query sequence has amino acids. The output of this pre-processing, i.e. the processed data, is in `processed/`.

| PFAM ID | Query sequence | Outgroup sequence | Taxa of outgroup
|---------|---------|-------|----------|
| PF00565 | SND1_HUMAN/552-660 | A0A060HE43_9ARCH/86-213 | Type of archaea
| PF00144 | A0A010Q9K6_9PEZI/15-292 | |
| PF02033 | Seq1239 | |

