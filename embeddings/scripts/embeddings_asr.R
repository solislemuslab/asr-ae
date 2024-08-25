library(tidyverse)
library(Rphylopars) #automatically loads ape
library(phytools)
library(network) # for plotting the tree as a network in embedding space

## Set global parameters #######
# Reconstructing for a real or simulated protein family?
REAL = FALSE
dir_name = if (REAL) "data/real/iqtree/tree_files" else "data/simulations/fast_trees"
N_str = "5k"
N = 5000
family = "COG438"

## Read in cluster categories ######
cluster_path = paste0(dir_name, "/", N_str, "/clusters/", family, "_clusters.txt")
cats <- read.table(cluster_path, header = TRUE) |>
  mutate(ClusterNumber = as.factor(ClusterNumber))
num_cats = length(levels(cats$ClusterNumber))
# Create vector of colors for clusters
cols = c(rainbow(num_cats), "black")
names(cols) = as.character(c(1:(num_cats-1), -1, 0 ))

## Read in family tree #####
if (!REAL) {
  tree_file = paste0(family, ".sim.trim.tree_cleaned")
  tree_path = paste(dir_name, N_str, tree_file, sep = "/")
  tree <- read.tree(tree_path)
  stopifnot(
    # check that tip labels are what they should be
    all(sort(paste0("N", 1:N)) == sort(tree$tip.label))
  )
  tree <- makeNodeLabel(tree)
}


## Read in embeddings #####
model = "model_ld2_wd0.01_epoch100_2024-08-19"
file = paste0(model, "_embeddings.csv")
msa = paste0(family, "-l150-s1")
embed_path = paste("embeddings", msa, file, sep="/")
embeds <- read.csv(embed_path) |>
  dplyr::rename(species = id)
stopifnot(
  # There should be no exact duplicates of embeddings
  all.equal(dplyr::distinct(embeds, dim0, dim1, .keep_all = TRUE), embeds)
)


# Because we dropped duplicate sequences prior to obtaining embeddings,
# we don't have embeddings for every tip in our tree
# We subset the tree to only include the tips that we have embeddings for.
subtree = keep.tip(tree, embeds$species)
# We also remove any extremely short external branches
degree(subtree) # check tree
thresh = .001
twig_ids <- subtree$edge[subtree$edge.length  < thresh & subtree$edge[,2] <= Ntip(subtree), 2]
subtree <- drop.tip(subtree, twig_ids)
degree(subtree)
# Save the final tree and the names of sequences that appear in it
subtree_path = str_replace(tree_path, "cleaned", "processed")
names_path = paste("data/simulations/processed",
                   msa,
                   "final_seq_names.txt", sep = "/")
write.tree(subtree, subtree_path)
write(subtree$tip.label, names_path)


## Plot tree colored by category ####
sim.tree = subtree
for (k in levels(cats$ClusterNumber)) {
  tips = filter(cats, ClusterNumber == k) |>
    pull(SequenceName)
  tip_idxs = na.omit(match(tips, sim.tree$tip.label))
  if (length(tip_idxs) > 0)
    sim.tree = paintBranches(sim.tree, edge=tip_idxs,
                            state = k, anc.state = "0")
}
plot(sim.tree,
     colors = cols,
     ftype = "off",
     lwd = .3
)

## Plot embeddings colored by category ####
embeds <- dplyr::left_join(embeds, cats, dplyr::join_by(species == SequenceName))
embeds <- embeds[match(subtree$tip.label,embeds$species),]
ntips = nrow(embeds)
ggplot(embeds, aes(dim0, dim1, col = ClusterNumber)) +
  geom_point(size = .8, alpha = .5) +
  scale_color_manual(values = cols) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))



## Fit model and get predicted reconstructed ancestral embeddings #####
# fit model
p_BM <- phylopars(trait_data = select(embeds, -c(ClusterNumber)),
                  tree = subtree)
# get dataframe including embeddings at both tips and reconstructed embeddings at ancestors
all_embeds = p_BM$anc_recon

# assign rownames that correspond to the ones in the original MSA
anc_rownames = as.integer(str_extract(subtree$node.label, "\\d+")) + N
rownames(all_embeds)[(ntips+1):nrow(all_embeds)] = anc_rownames

# save reconstructed ancestral embeddings
anc_embed_path = str_match(embed_path, "^(.*)embeddings.csv")[2] |>
  str_c("anc-embeddings.csv")
anc_embeds = all_embeds[(ntips+1):nrow(all_embeds),]
write.csv(anc_embeds, anc_embed_path)


# Plot as a network
net = as.network(subtree)
vert_type = rep(c("tip", "internal"),
                  c(ntips, nrow(all_embeds)-ntips))
vert_col = c(#root = "green",
  internal = scales::alpha("blue", .7),
  tip = scales::alpha("red", .7))
edge_col = scales::alpha("black", .2)
p <- plot.network(net,
                  coord = all_embeds,
                  vertex.border = NA,
                  vertex.cex = .6,
                  edge.lwd = .25,
                  edge.col = edge_col,
                  usearrows = FALSE,
                  vertex.col = vert_col[vert_type],
                  xlab = "Dimension 1",
                  ylab = "Dimension 2",
                  #label.cex = .3,
                  #boxed.labels = T,
                  #displaylabels = T,
                  #label.pos = 5,
                  suppress.axes = FALSE,
)
legend("bottomleft", inset = c(.075,.075),
       pch = 16,
       cex = .7,
       border = "black",
       bty = "n",
       col = vert_col,
       legend = names(vert_col))











