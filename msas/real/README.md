For PFAM protein family 00565 (Staphylococcal nuclease homologues), which was analyzed by [Ding](https://www.nature.com/articles/s41467-019-13633-0), we downloaded the raw "full" MSA from the Interpro website on date x and filtered this alignment to retain only sequences from Eukaryotic species.

For PFAM protein family PF00144 (beta-lactamase), which was analyzed by [Detlefsen](https://www.nature.com/articles/s41467-022-29443-w), we download `https://github.com/MachineLearningLifeScience/meaningful-protein-representations/blob/master/tape/PF00144_full.txt`, which we assume was obtained from the full alignment of the family from Interpro.

Finally, for PF02033 (Ribosome-binding factor A), we download an alignment the alignment found at `aln/1josA.aln` of `http://bioinf.cs.ucl.ac.uk/downloads/PSICOV/suppdata/`, which was published with the [PSICOV paper](https://academic.oup.com/bioinformatics/article/28/2/184/198108). We don't have meaningful sequence labels for this family.

To each of these MSAs, we then applied the pre-processing steps from the Ding paper by running the script `scripts/process_msa.py`, which requires specifying a "query" or "target" sequence (an older version of this script, used to generate older versions of the pre-processed data is made available as `scripts/process_msa-og.py`). For each family, the output of this pre-processing includes the data objects that our VAE can process and is placed in the `processed` directory.

| PFAM ID | Query sequence 
|---------|---------|
| PF00565 | SND1_HUMAN/552-660 
| PF00144 | A0A010Q9K6_9PEZI/15-292 
| PF02033 | Seq1239  

