# This script midpoint roots and cleans the unrooted FastTree benchmark study trees
# by removing short branches and replacing bootstrap support values with internal node identifiers.

suppressPackageStartupMessages({
    library(ape) 
    library(phytools)
    library(stringr)
})
trimming_thresh = 0.001
simulated_trees = list(
    "1250" = list.files(path = file.path("trees", "fast_trees", "1250"), pattern = "\\.sim.trim.tree$", full.names = TRUE),
    "5000" = list.files(path = file.path("trees", "fast_trees", "5000"), pattern = "\\.sim.trim.tree$", full.names = TRUE)
)
for (n_seq in names(simulated_trees)) {
    n_seqi = as.integer(n_seqi)
    for (tree_file in simulated_trees[[n_seq]]) {
        cat("Processing tree file: ", basename(tree_file), "\n")
        # read the tree
        tree <- read.tree(tree_file)
        # check that tree is what we expect
        stopifnot( 
        all(sort(paste0("N", 1:n_seqi)) == sort(tree$tip.label)),
        tree$Nnode == n_seqi-2
        )
        # midpoint root
        tree <- midpoint_root(tree)
        # Replace bootstrap supports with number which identify the internal nodes
        tree$node.label = paste0("A",(n_seqi+1):(2*n_seqi-1)) #e.g.A5001 -> A9999
        # Replace all negative branch lengths with 0
        tree$edge.length[tree$edge.length < 0] <- 0
        # Trim extremely short external branches 
        twig_ids <- tree$edge[tree$edge.length < trimming_thresh & tree$edge[, 2] <= Ntip(tree), 2] 
        cleaned_tree <- drop.tip(tree, twig_ids)
        # Report and write
        cat("Cleaned tree has ", Ntip(cleaned_tree), " tips and ", Nnode(cleaned_tree), " internal nodes.\n")
        cleaned_tree_path <- str_replace(tree_file, "sim.trim", "clean") 
        write.tree(cleaned_tree, cleaned_tree_path)
     }
}

## real pf00565 tree
fp <- file.path("trees", "inferred_real_trees", "PF00565.treefile")
outgroup_tip <- "A0A060HE43_9ARCH_86-213"
tree <- read.tree(fp)
# root tree with outgroup
if (!outgroup_tip %in% tree$tip.label) {
        stop("Outgroup tip not found in tree: ", outgroup_tip)
    }
tree <- root(tree, outgroup = outgroup_tip, resolve.root = TRUE)
# Replace the last "_" in the tip names with "/"
tree$tip.label <- str_replace(tree$tip.label, "(.*)_([^_]*)$", "\\1/\\2")
# Provide labels for the internal nodes
n_leaf <- length(tree$tip.label)
tree$node.label = paste0("A",(n_leaf+1):(2*n_leaf-1)) 
cleaned_tree_path <- str_replace(fp, ".treefile", "_rooted.treefile")
write.tree(tree, cleaned_tree_path)


## real PF00144 tree
fp <- file.path("trees", "inferred_real_trees", "PF00144.treefile")
tree <- read.tree(fp)
# Replace the last "_" in the tip names with "/"
tree$tip.label <- str_replace(tree$tip.label, "(.*)_([^_]*)$", "\\1/\\2")
n_leaf <- length(tree$tip.label)
tree$node.label = paste0("A",(n_leaf+1):(2*n_leaf-2)) 
cleaned_tree_path <- str_replace(fp, ".treefile", "_cleaned.treefile")
write.tree(tree, cleaned_tree_path)


