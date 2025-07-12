These trees are available from http://www.microbesonline.org/fasttree/downloads/aa5K_new.tar.gz (trees with 5k leaves) and http://www.microbesonline.org/fasttree/downloads/aa1250.tar.gz (trees with 1250 leaves). The original Fasttree paper explains how these trees were estimated based on alignments of protein families from the Clusters of Orthologous Groups (COG), using either PhyML (for the 1250 leaf trees) or FastTree (for the 5k leaf trees). Note that the FastTree algorithm included some negative external branch lengths in the 5k leaf trees. 

We clean these trees with...

We cluster these trees using TreeCluster (Balaban et al) with their Max method. For example, for `COG277.clean.tree` we run:
```
 TreeCluster.py -i trees/fast_trees/1250/COG277.clean.tree -o trees/fast_trees/1250/COG277_clusters.txt -m MAX -t 3
```