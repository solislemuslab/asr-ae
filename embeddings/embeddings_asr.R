# This R script performs ancestral state reconstruction (ASR) of the sequence embeddings in a given MSA, and writes the
# "reconstructed" embeddings of the internal nodes of the tree to a CSV file. It also plots the tree in the embedding space.
# It's not yet implemented for the real data, but it works for the simulated data.

suppressPackageStartupMessages({
    library(fs)
    library(tidyverse)
    library(Rphylopars) # automatically loads ape
    library(phytools)
    library(network) # for plotting the tree as a network in embedding space
})

# get command line arguments
data_path <- commandArgs(trailingOnly = TRUE)[1]
model <- commandArgs(trailingOnly = TRUE)[2]

# file path wrangling
msa_id <- path_file(data_path)
family <- str_extract(msa_id, "^[a-zA-Z0-9]+")
sim <- str_detect(msa_id, "COG")

# tree directory
if (sim) {
    n_seq <- path_file(path_dir(data_path))
    # get the type of simulation, either independent or coupling
    sim_type <- path_split(data_path)[[1]][2]
    tree_dir <- path("trees", "fast_trees", n_seq)
} else {
    tree_dir <- path("trees", "inferred_real_trees")
}

# plot directory
plot_dir <- if (sim) path("plots", sim_type, n_seq, msa_id) else path("plots", "real", msa_id)

# cluster file
# cluster_path = path(tree_dir, "clusters", paste0(family, "_clusters.txt"))
# cats <- read.table(cluster_path, header = TRUE) |>
#   mutate(ClusterNumber = as.factor(ClusterNumber))
# num_cats = length(levels(cats$ClusterNumber))
# # create vector of colors for clusters
# cols = c(rainbow(num_cats), "black")
# names(cols) = as.character(c(1:(num_cats-1), -1, 0 ))

## Read in family tree #####
if (sim) {
    tree_file <- paste0(family, ".sim.trim.tree_cleaned")
    tree_path <- path(tree_dir, tree_file)
    tree <- read.tree(tree_path)
    stopifnot(
        # check that tip labels are what they should be
        all(sort(paste0("N", 1:n_seq)) == sort(tree$tip.label))
    )
    tree <- makeNodeLabel(tree)
}

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
stopifnot(
    # There should be no exact duplicates of embeddings
    all.equal(dplyr::distinct(embeds, dim0, dim1, .keep_all = TRUE), embeds)
)

# Because we dropped duplicate sequences prior to obtaining embeddings, we don't have embeddings for every tip in our tree
# We subset the tree to only include the tips that we have embeddings for.
subtree <- keep.tip(tree, embeds$species)
# We perform additional pruning, remove any extremely short external branches, and update the embeddings dataframe accordingly
# This is done to avoid numerical issues when running phylopars()
thresh <- .001
twig_ids <- subtree$edge[subtree$edge.length < thresh & subtree$edge[, 2] <= Ntip(subtree), 2]
subtree <- drop.tip(subtree, twig_ids)
embeds <- embeds[match(subtree$tip.label, embeds$species), ]
ntips <- Ntip(subtree)
cat("Pruned tree has ", Ntip(subtree), " tips and ", Nnode(subtree), " internal nodes.\n")
# Save the final tree and the names of sequences that appear in it
subtree_path <- str_replace(tree_path, "cleaned", "revised")
names_path <- path(data_path, "final_seq_names.txt")
write.tree(subtree, subtree_path)
write(subtree$tip.label, names_path)


## Plot tree colored by category ####
# sim.tree = subtree
# for (k in levels(cats$ClusterNumber)) {
#   tips = filter(cats, ClusterNumber == k) |>
#     pull(SequenceName)
#   tip_idxs = na.omit(match(tips, sim.tree$tip.label))
#   if (length(tip_idxs) > 0)
#     sim.tree = paintBranches(sim.tree, edge=tip_idxs,
#                             state = k, anc.state = "0")
# }
# png(file = path(plot_dir, paste0(model, "_colored-tree.png")), width = 1200, height = 1200)
# plot(sim.tree,
#      colors = cols,
#      ftype = "off",
#      lwd = 1
# )
# invisible(dev.off())

## Plot embeddings colored by category ####
# embeds <- dplyr::left_join(embeds, cats, dplyr::join_by(species == SequenceName))
# colored_embeds = ggplot(embeds, aes(dim0, dim1, col = ClusterNumber)) +
#   geom_point(size = .8, alpha = .5) +
#   scale_color_manual(values = cols) +
#   theme_minimal() +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.line = element_line(colour = "black"))
# ggsave(
#   path(plot_dir, paste0(model, "_colored-embeddings.png")),
#   colored_embeds,
#   bg="white",
#   width = 6,
#   height = 6
#   )

## Fit model and get predicted reconstructed ancestral embeddings #####
# fit model
# p_BM <- phylopars(trait_data = select(embeds, -c(ClusterNumber)),
#                   tree = subtree)
p_BM <- phylopars(trait_data = embeds, tree = subtree)
# get dataframe including embeddings at both tips and reconstructed embeddings at ancestors
all_embeds <- p_BM$anc_recon

# assign rownames that correspond to the ones in the original MSA
anc_rownames <- as.integer(str_extract(subtree$node.label, "\\d+")) + as.integer(n_seq)
rownames(all_embeds)[(ntips + 1):nrow(all_embeds)] <- anc_rownames

# save reconstructed ancestral embeddings
anc_embed_path <- str_match(embed_path, "^(.*)embeddings.csv")[2] |>
    str_c("anc-embeddings.csv")
anc_embeds <- all_embeds[(ntips + 1):nrow(all_embeds), ]
write.csv(anc_embeds, anc_embed_path)

# Plot tree in (in first two dimensions of) embedding space
net <- as.network(subtree)
vert_type <- rep(
    c("tip", "internal"),
    c(ntips, nrow(all_embeds) - ntips)
)
vert_col <- c( # root = "green",
    internal = scales::alpha("blue", .7),
    tip = scales::alpha("red", .7)
)
edge_col <- scales::alpha("black", .2)
png(file = path(plot_dir, paste0(model, "_network.png")), width = 1200, height = 1200)
plot.network(net,
    coord = all_embeds[, 1:2],
    xlim = range(all_embeds[, 1]),
    ylim = range(all_embeds[, 2]),
    vertex.border = NA,
    vertex.cex = .3,
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
legend("bottomleft",
    pch = 16,
    cex = 2,
    border = "black",
    bty = "n",
    col = vert_col,
    legend = names(vert_col)
)
invisible(dev.off())
