# ASR using RAxML-NG for real datasets provided by AIRAS [https://github.com/bryankolaczkowski/airas/tree/master/data/real_sequences]

cd: /Users/haileylouw/Desktop/asr-ae/data/raxml-ng

# CARD1 with JTT+G evolutionary model

./raxml-ng  --ancestral --msa ./data/structaligned_card1.fasta --tree ./data/brlens_and_labels_card1.tre --msa-format FASTA --data-type AA --model LG4M

# DRSM1

./raxml-ng --ancestral --msa ./data/structaligned_drsm1.fasta --tree ./data/brlens_and_labels_drsm1.tre --msa-format FASTA --data-type AA --model LG4M

# DRSM2

# Build Tree with MUSCLE Alignment in IQ Tree

cd: /Users/haileylouw/Desktop/asr-ae/data/iqtree/bin

IQ-Tree with nonparametric bootstrap:  `./iqtree2 -s /Users/haileylouw/Desktop/asr-ae/data/raxml-ng/data/drsm2_aligned_MUCSLE.fasta -m MFP -m WAG+G+I  -b 100`

IQ-Tree with ultrafast boostrap: `./iqtree2 -s /Users/haileylouw/Desktop/asr-ae/data/raxml-ng/data/drsm2_aligned_MUCSLE.fasta -m MFP -m WAG+G+I  -B 1000`

# ASR for Sequence Alignment with MUSCLE & Consensus Tree Used 

cd: /Users/haileylouw/Desktop/asr-ae/data/raxml-ng

`./raxml-ng --ancestral --msa ./data/drsm2_aligned_MUCSLE.fasta --tree ./data/brlens_and_labels_drsm2.tre --msa-format FASTA --data-type AA --model WAG+G+I`

# ASR for Sequence Alignment with MUSCLE & IQ-Tree UFB Result

cd: /Users/haileylouw/Desktop/asr-ae/data/raxml-ng

Consensus Tree: 
`./raxml-ng --ancestral --msa ./data/drsm2_aligned_MUCSLE.fasta --tree .data/drsm2_UFB_tree_files/drsm2_aligned_MUCSLE.fasta.ufb.contree --msa-format FASTA --data-type AA --model WAG+G+I`

ML Tree: 
`./raxml-ng --ancestral --msa ./data/drsm2_aligned_MUCSLE.fasta --tree .data/drsm2_UFB_tree_files/drsm2_aligned_MUCSLE.fasta.ufb.treefile --msa-format FASTA --data-type AA --model WAG+G+I`

# ASR for Sequence Alignment with MUSCLE & IQ-Tree Nonparametric Boostrap Result

cd: /Users/haileylouw/Desktop/asr-ae/data/raxml-ng

Consensus Tree: 
`./raxml-ng --ancestral --msa ./data/drsm2_aligned_MUCSLE.fasta --tree ./data/drsm2_bootstrap_tree_files/drsm2_aligned_MUCSLE.fasta.boot.contree --msa-format FASTA --data-type AA --model WAG+G+I`

ML Tree: 
`./raxml-ng --ancestral --msa ./data/drsm2_aligned_MUCSLE.fasta --tree ./data/drsm2_bootstrap_tree_files/drsm2_aligned_MUCSLE.fasta.boot.treefile --msa-format FASTA --data-type AA --model WAG+G+I`

# DRSM3

./raxml-ng --ancestral --msa ./data/structaligned_drsm3.fasta --tree ./data/brlens_and_labels_drsm3.tre --msa-format FASTA --data-type AA --model LG4M

# RD1

./raxml-ng --ancestral --msa ./data/structaligned_rd1.fasta --tree ./data/brlens_and_labels_rd1.tre --msa-format FASTA --data-type AA --model LG4M

