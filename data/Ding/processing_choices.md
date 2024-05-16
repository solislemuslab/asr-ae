For each PFAM protein family, we downloaded the raw "full" MSA from the Interpro website. We then pre-processed this MSA. First we filtered to retain only sequences from Eukaryotic species and at most 50K of them + 1 sequence from a non-Eukaryotic species as an outgroup. Then, we applied the pre-processing steps from the Ding paper to make the MSA nicer (get rid a lot of the gaps). This pre-processing requires specifying a query sequence. For the PF00041 family, we used the same sequence that Ding used as the query sequence. For the other two families, we do not know what Ding used so we chose our own query sequences (for PF00067, we pick a query sequence that will not lead to dropping too many sequences during the pre-processing).

| PFAM ID | Query sequence | Outgroup sequence | Taxa
|---------|---------|-------|----------|
| PF00565 | SND1_HUMAN/552-660 | A0A060HE43_9ARCH/86-213 | Type of archaea
| PF00041 | TENA_HUMAN/804-884 | A0A023BR93_9FLAO/757-833 | Aquimarina atlantica (type of bacteria)
| PF00067 | A0A8J5V3X2_ZIZPA/310-424 | A0A010Z6J1_9ACTN/277-406 | Type of bacteria
| PF00144 | 0A010Q9K6_9PEZI/15-292 | |

