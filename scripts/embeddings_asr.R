library(ape) # reads tree
library(mvMORPH) # has multivariate trait evolutionary models
library(stringr) # string manipulating
library(castor) # for subsetting trees
library(network) # for plotting the tree as a network in embedding space
library(scales) # for plotting transparency

# read in inferred tree
# check that tip labels are in format "X_Y_Z"
# replace second underscore with slash to match embedding names
pf00565.tree <- read.tree("data/iqtree/tree_files/pf00565_rerooted.tree")
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
# get rid of identical sequences or equivalently identical embeddings
pf00565.subdata = dplyr::distinct(pf00565.data)
subtree_extract = get_subtree_with_tips(pf00565.tree,
                                        only_tips = rownames(pf00565.subdata))
pf00565.subtree = subtree_extract$subtree |> reorder()
pf00565.subdata <- pf00565.subdata[pf00565.subtree$tip.label,]

# fit gls model
pf00565.matrix <- data.matrix(pf00565.subdata)
pf00565.list <- list(pf00565.matrix=pf00565.matrix)
fit.pf00565 <- mvgls(pf00565.matrix~1,
                     data=pf00565.list,
                     pf00565.subtree,
                     model = "BM",
                     method = "LL")

# show estimates of diffusion matrix
fit.pf00565$sigma
fit.pf00565$logLik

# can also try to use mvBM(), followed by estim(tree, data, fit, asr = TRUE)
# However, the following code takes forever to run??
#fit.bm.pf00565 <- mvBM(pf00565.subtree, pf00565.subdata, model = "BM1")

# get ancestral states
anc <- ancestral(fit.pf00565)
all_embeddings = rbind(pf00565.subdata, anc)
vertex_names = rownames(all_embeddings)
vertex_type = ifelse(str_detect(vertex_names,"node"), "internal", "tip")
vertex_type[pf00565.subtree$root] = "root"

# Plot as a network
net = as.network(pf00565.subtree)
colors = c(root = "green",
           internal = alpha("blue",.4),
           tip = alpha("orange", .4))
p <- plot.network(net,
             coord = all_embeddings,
             vertex.cex = 1.5,
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
legend("bottomleft", inset = c(.01,.01),
       pch = 1,
       cex = 1,
       col = "black",
       bty = "n",
       legend = names(colors) )
grid()
# weirdly, the root is not being inferred to be in the center of the space


#######################
# Next, try out castor

castor <- fit_bm_model(pf00565.subtree, pf00565.matrix, Nbootstraps = 100)
castor$diffusivity # same estimated rate matrix up to a scaling as mvMORPH
castor$loglikelihood # same log-likelihood

asr_scp1 <- asr_squared_change_parsimony(pf00565.tree, pf00565.matrix[,1])
asr_scp1

anc <- ancestral(fit.pf00565)
all_embeddings = rbind(pf00565.subdata, anc)
vertex_names = rownames(all_embeddings)
vertex_type = ifelse(str_detect(vertex_names,"node"), "internal", "tip")
vertex_type[pf00565.subtree$root] = "root"

# Plot as a network
net = as.network(pf00565.subtree)
colors = c(root = "green",
           internal = alpha("blue",.4),
           tip = alpha("orange", .4))
p <- plot.network(net,
                  coord = all_embeddings,
                  vertex.cex = 1.5,
                  usearrows = FALSE,
                  vertex.col = colors[vertex_type],
                  xlab = "Dimension 1",
                  ylab = "Dimension 2",
                  xlim = c(-.5, .5),
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
       legend = names(colors) )











