library(tidyverse)
library(Rphylopars) #automatically loads ape
library(phytools)
library(network) # for plotting the tree as a network in embedding space

# Reconstructing for a real or simulated protein family?
REAL = FALSE
dir_name = if (REAL) "data/real/iqtree/tree_files" else "data/simulations/fast_trees"
N_str = "1250"
N = 1250
family = "COG154"
# Read in cluster categories
cluster_path = paste0(dir_name, "/", N_str, "/clusters/", family, "_clusters.txt")
cats <- read.table(cluster_path, header = TRUE) |>
  mutate(ClusterNumber = as.factor(ClusterNumber))
num_cats = length(levels(cats$ClusterNumber))

# Read in family tree
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
sim.tree = tree
for (k in levels(cats$ClusterNumber)) {
  tips = filter(cats, ClusterNumber == k) |>
    pull(SequenceName)
  sim.tree = paintBranches(sim.tree, edge=match(tips, sim.tree$tip.label),
                       state = k, anc.state = "0")
}
cols = c(rainbow(num_cats), "black")
names(cols) = as.character(c(1:(num_cats-1), -1, 0 ))
plot(sim.tree,
     colors = cols,
     ftype = "off",
     lwd = .7
     )


# Read in embeddings
model = "model_ld2_wd0_epoch30_2024-08-01_embeddings.csv"
msa = paste0(family, "-l150-s0.5")
embed_path = paste("embeddings", msa, model, sep="/")
embeds <- read.csv(embed_path) |>
  dplyr::rename(species = id)
stopifnot(
  # There should be no exact duplicates of embeddings
  all.equal(dplyr::distinct(embeds, dim0, dim1, .keep_all = TRUE), embeds)
)
# join with cluster categories
embeds <- dplyr::left_join(embeds, cats, dplyr::join_by(species == SequenceName))
ggplot(embeds, aes(dim0, dim1, col = ClusterNumber)) +
  geom_point() +
  scale_color_manual(values = cols) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))


# Might have to subset embeds to get Brownian motion model to fit
# set.seed(10)
# embeds <- embeds |>
#   dplyr::slice_sample(n = 1000)


# Because we dropped duplicate sequences prior to obtaining embeddings,
# we don't have embeddings for every tip in our tree
# We subset the tree to only include the tips that we have embeddings for.
# We then enforce that order of rows in the embeddings dataframe matches the order in the tree
subtree = keep.tip(tree, embeds$species)
embeds <- embeds[match(subtree$tip.label,embeds$species),]
ntips = nrow(embeds)

# Fit model and get predicted reconstructed ancestral embeddings
p_BM <- phylopars(trait_data = select(embeds, -ClusterNumber),tree = subtree)
all_embeds = p_BM$anc_recon

# assign rownames that correspond to the ones in the original MSA
anc_rownames = as.integer(str_extract(subtree$node.label, "\\d+")) + 1250
rownames(all_embeds)[(ntips+1):nrow(all_embeds)] = anc_rownames

# save reconstructed ancestral embeddings
anc_embed_path = str_match(embed_path, "^(.*)embeddings.csv")[2] |>
  str_c("anc-embeddings.csv")
anc_embeds = all_embeds[(ntips+1):nrow(all_embeds),]
write.csv(anc_embeds, anc_embed_path)


# Plot as a network
net = as.network(subtree)
vertex_type = rep(c("tip", "internal"),
                  c(ntips, nrow(all_embeds)-ntips))
colors = c(#root = "green",
  internal = scales::alpha("blue", 1),
  tip = scales::alpha("red", 1))
p <- plot.network(net,
                  coord = all_embeds,
                  vertex.border = NA,
                  usearrows = FALSE,
                  vertex.col = colors[vertex_type],
                  xlab = "Dimension 1",
                  ylab = "Dimension 2",
                  #label.cex = .3,
                  #boxed.labels = T,
                  #displaylabels = T,
                  #label.pos = 5,
                  suppress.axes = FALSE
)
legend("bottomleft", inset = c(.01,.01),
       pch = 16,
       cex = 1,
       border = "black",
       bty = "n",
       col = colors,
       legend = names(colors))











