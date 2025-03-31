# This R script performs ancestral state reconstruction (ASR) of the sequence embeddings in a given MSA, and writes the
# "reconstructed" embeddings of the internal nodes of the tree to a CSV file. It also plots the tree in the embedding space.
# It's not yet implemented for the real data, but it works for the simulated data.

suppressPackageStartupMessages({
    library(Biostrings) # for reading and writing fasta 
    library(fs) # make sure this is loaded after Biostrings to keep path() function from fs
    library(tidyverse)
    library(Rphylopars) # automatically loads ape
    library(phytools)
    library(network) # for plotting the tree as a network in embedding space
})

## Function to filter fasta file #####
filter_fasta <- function(og_path, new_path, keep) {
    # Read the original fasta file
    fasta <- readAAStringSet(og_path, format = "fasta")
    # Filter sequences based on the 'keep' list
    filtered_fasta <- fasta[names(fasta) %in% keep]
    # Write the filtered sequences to a new fasta file
    writeXStringSet(filtered_fasta, new_path, format = "fasta")
}

# get command line arguments
data_path <- commandArgs(trailingOnly = TRUE)[1]
model <- commandArgs(trailingOnly = TRUE)[2]

# file path wrangling
msa_id <- path_file(data_path)
family <- str_extract(msa_id, "^[a-zA-Z0-9]+")
sim <- str_detect(msa_id, "COG")

# tree directory
if (sim) {
    n_seq <- as.integer(path_file(path_dir(data_path)))
    # get the type of simulation, either independent, coupling, or potts
    sim_type <- path_split(data_path)[[1]][2]
    tree_dir <- path("trees", "fast_trees", n_seq)
} else {
    tree_dir <- path("trees", "inferred_real_trees")
}

# plot directory
plot_dir <- if (sim) path("plots", sim_type, n_seq, msa_id) else path("plots", "real", msa_id)

## Read in family tree #####
if (sim) {
    tree_file <- paste0(family, ".sim.trim.tree")
    tree_path <- path(tree_dir, tree_file)
    tree <- read.tree(tree_path)
    stopifnot(
        # check that tip labels are what they should be
        all(sort(paste0("N", 1:n_seq)) == sort(tree$tip.label))
    )
} else {
     tree_file <- paste0(family, ".treefile")
}

tree$node.label = paste0("Node", (n_seq+1):(2*n_seq-2)) #e.g. Node5001 -> Node9998

## Read in embeddings #####
embed_dir <- path("embeddings", "data")
if (sim) {
    embed_path <- path(
        embed_dir,
        sim_type,
        n_seq,
        msa_id,
        paste0(model, "_embeddings.csv")
    )
} else {
    embed_path <- path(
        embed_dir,
        "real",
        msa_id,
        paste0(model, "_embeddings.csv")
    )
}
embeds <- read.csv(embed_path) |>
    dplyr::rename(species = id)

# Before running phylopars() to fit a Brownian motion model and perform ASR, 
# we trim our tree to get rid of extremely short external branches, which cause problems for the function 
# This effectively removes some leaf nodes
thresh <- 0.001
# get ids of the tips of the short external branches
twig_ids <- tree$edge[tree$edge.length < thresh & tree$edge[, 2] <= Ntip(tree), 2] 
# drop from tree
trimmed_tree <- drop.tip(tree, twig_ids)
# drop from embeddings
embeds <- embeds[match(trimmed_tree$tip.label, embeds$species), ]
# print number of tips and internal nodes in the pruned tree
cat("Pruned tree has ", Ntip(trimmed_tree), " tips and ", Nnode(trimmed_tree), " internal nodes.\n")
ntips_trimmed <- Ntip(trimmed_tree)
# Save names of leaves in the final data set
names_path <- path(data_path, "final_seq_names.txt")
write(trimmed_tree$tip.label, names_path)
# Save filtered tree
trimmed_tree_path <- str_replace(tree_path, "trim", "fully_trim") #fix this for real trees
write.tree(trimmed_tree, trimmed_tree_path)
# Save filtered MSA
fasta_path <- path(data_path, "seq_msa_char.fasta")
filter_fasta(fasta_path, fasta_path, trimmed_tree$tip.label)


## Perform ancestral state reconstruction
p_BM <- phylopars(trait_data = embeds, tree = trimmed_tree)
# get dataframe including embeddings at both tips and reconstructed embeddings at ancestors
all_embeds <- p_BM$anc_recon 
# note that all_embeds has the first ntips rows corresponding to tips and the rest to internal nodes
# the ordering is the same as the ordering of the tips and internal nodes in the tree object 
# we add rownames to all_embeds that correspond to the names for internal nodes in the original simulated MSA, 
# which corresponds to a preorder traversal
anc_rownames <- as.integer(str_extract(trimmed_tree$node.label, "\\d+")) 
rownames(all_embeds)[(ntips_trimmed + 1):nrow(all_embeds)] <- anc_rownames

# save reconstructed ancestral embeddings
anc_embed_path <- str_match(embed_path, "^(.*)embeddings.csv")[2] |>
    str_c("anc-embeddings.csv")
anc_embeds <- all_embeds[(ntips_trimmed + 1):nrow(all_embeds), ]
write.csv(anc_embeds, anc_embed_path)

# Plot tree in (in first two dimensions of) embedding space
net <- as.network(trimmed_tree)
vert_type <- rep(
    c("tip", "internal"),
    c(ntips_trimmed, nrow(all_embeds) - ntips_trimmed)
)
vert_col <- c( # root = "green",
    internal = scales::alpha("blue", .4),
    tip = scales::alpha("red", .4)
)
vert_size <- c( 
    internal = .3,
    tip = .3
)
edge_col <- scales::alpha("black", .2)
png(file = path(plot_dir, paste0(model, "_network.png")), width = 1200, height = 1200)
if (ncol(all_embeds) > 2) {
    pca <- prcomp(all_embeds)
    plotted_coords <- pca$x[, 1:2]
} else {
   plotted_coords <- all_embeds
}
    

plot.network(net,
    coord = plotted_coords[, 1:2],
    xlim = range(plotted_coords[, 1]),
    ylim = range(plotted_coords[, 2]),
    vertex.border = NA,
    vertex.cex = vert_size[vert_type],
    edge.lwd = .3,
    edge.col = edge_col,
    usearrows = FALSE,
    vertex.col = vert_col[vert_type],
    # label.cex = .3,
    # boxed.labels = T,
    # displaylabels = T,
    # label.pos = 5,
    suppress.axes = TRUE,
)
# highlight the root, which should be the first row of the internal nodes in all_embeds
root_index <- ntips_trimmed + 1
points(plotted_coords[root_index, 1], plotted_coords[root_index, 2], 
       col = "green", pch = 16, cex = 1)

legend("bottomleft",
    pch = 16,
    cex = 2,
    border = "black",
    bty = "n",
    col = vert_col,
    legend = names(vert_col)
)
invisible(dev.off())
