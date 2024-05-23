library(ape) # reads tree
library(mvMORPH) # has multivariate trait evolutionary models
library(stringr) # string manipulating
library(castor) # for subsetting trees
library(network) # for plotting the tree as a network in embedding space
library(scales) # for plotting transparency

# read in inferred tree
# check that tip labels are in format "X_Y_Z"
# replace second underscore with slash to match embedding names
pf00565.tree <- read.tree("data/iqtree/tree_files/seq_msa_char_pf00565.fasta.treefile")
local({
  tip_name_pieces = str_split(pf00565.tree$tip.label, "_")
  num_pieces = sapply(tip_name_pieces, length)
  stopifnot(all(num_pieces == 3))
})
pf00565.tree$tip.label <- str_replace(pf00565.tree$tip.label,
                                     "_([^_]*)_",
                                     "_\\1/")

# read in embedding
pf00565.data <- read.csv("embeddings/PF00565/model_2024-05-06_embeddings.csv",
                         row.names=1, stringsAsFactors = TRUE)

# let's start by just analyzing a subset of the tree with n tips
n = 50
set.seed(100)
subset = sample(pf00565.tree$tip.label, n)
subtree_extract = get_subtree_with_tips(pf00565.tree, only_tips = subset)
pf00565.subtree = subtree_extract$subtree
pf00565.subtree = reorder(pf00565.subtree)
pf00565.data <- pf00565.data[pf00565.subtree$tip.label,]

# fit gls model
pf00565.matrix <- data.matrix(pf00565.data)
pf00565.list <- list(pf00565.matrix=pf00565.matrix)
fit.pf00565 <- mvgls(pf00565.matrix~1,
                     data=pf00565.list,
                     pf00565.subtree,
                     model = "BM",
                     method = "LL")
anc <- ancestral(fit.pf00565)
all_embeddings = rbind(pf00565.data, anc)
vertex_names = rownames(all_embeddings)
vertex_type = ifelse(str_detect(vertex_names,"node"), "internal", "tip")
vertex_type[pf00565.subtree$root] = "root"

# Plot as a network
net = as.network(pf00565.subtree)
colors = c(root = alpha("red", 1),
           internal = alpha("blue",.4),
           tip = alpha("orange", .4))
p <- plot.network(net,
             coord = all_embeddings,
             vertex.cex = 1.5,
             usearrows = FALSE,
             vertex.col = colors[vertex_type],
             xlab = "Dimension 1",
             ylab = "Dimension 2",
             xlim = c(-5,5),
             ylim = c(-5, 5),
             #label.cex = .3,
             #boxed.labels = T,
             #displaylabels = T,
             #label.pos = 5,
             suppress.axes = FALSE,
             )
legend("bottomleft", inset = c(.1,.1),
       pch = 16,
       cex = 1,
       border = "black",
       bty = "n",
       col = colors,
       legend = names(colors))
legend("bottomleft", inset = c(.1,.1),
       pch = 1,
       cex = 1,
       col = "black",
       bty = "n",
       legend = names(colors),
       add )

# Next, try out castor::fit_bm_model()




