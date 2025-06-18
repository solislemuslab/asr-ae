# This script midpoint roots and cleans the unrooted FastTree benchmark study trees
# by removing short branches and replacing bootstrap support values with internal node identifiers.

suppressPackageStartupMessages({
    library(ape) 
    library(phytools)
    library(stringr)
})
trimming_thresh = 0.001
for (n_seq in c(1250, 5000)) {
    for (tree_file in list.files(path = file.path("trees", "fast_trees", n_seq), pattern = "\\.sim.trim.tree$", full.names = TRUE)) {
        cat("Processing tree file: ", basename(tree_file), "\n")
        # read the tree
        tree <- read.tree(tree_file)
        # check that tree is what we expect
        stopifnot( 
        all(sort(paste0("N", 1:n_seq)) == sort(tree$tip.label)),
        tree$Nnode == n_seq-2
        )
        # midpoint root
        tree <- midpoint_root(tree)
        # Replace bootstrap supports with number which identify the internal nodes
        tree$node.label = paste0("A",(n_seq+1):(2*n_seq-1)) #e.g.A5001 -> A9999
        # Replace all negative branch lengths with 0
        tree$edge.length[tree$edge.length < 0] <- 0
        # Trim extremely short external branches 
        twig_ids <- tree$edge[tree$edge.length < trimming_thresh & tree$edge[, 2] <= Ntip(tree), 2] 
        cleaned_tree <- drop.tip(tree, twig_ids)
        # Report and write
        cat("Cleaned tree has ", Ntip(cleaned_tree), " tips and ", Nnode(cleaned_tree), " internal nodes.\n")
        cleaned_tree_path <- str_replace(tree_file, "sim.trim", "clean") #fix this for real trees
        write.tree(cleaned_tree, cleaned_tree_path)
     }
}