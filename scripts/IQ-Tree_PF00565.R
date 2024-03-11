## IQ-Tree Script for ASR of PF0065 ## 

# connect to WID server 

`ssh hlouw@solislemus-001.discover.wisc.edu`

# IQ-Tree folder 

`cd hlouw@solislemus-001.discovery.wisc.edu/mnt/ws/home/hlouw/iqtree-project`

# IQ-Tree command with 1000 Ultrafast Bootstrap replicates

`./iqtree-1.6.12-Linux/bin/iqtree -s /mnt/ws/home/hlouw/iqtree-project/PF00565_full-seq_msa_char.fasta -m MFP -m LG+G20 -bb 1000`