library(fs) # make sure this is loaded after Biostrings to keep path() function from fs
library(ape)
library(phytools)
library(stringr)

data_path = "msas/real/processed/PF00144"
labeled_seq_file = "msas/real/raw/PF00144_full_length_sequences_labeled.fasta"
model = "ding_layers1000_ld2_wd0.0_epoch500_2025-08-12.pt"

# name and path wrangling
if (str_detect(model, "\\.pt$")) {
    model <- str_remove(model, "\\.pt$")
}
msa_id <- path_file(data_path)
family <- str_extract(msa_id, "^[a-zA-Z0-9]+")
sim <- !str_detect(data_path, "real")
plot_dir <- if (sim) path("plots", sim_type, n_seq, msa_id) else path("plots", "real", msa_id)

## Read in family tree #####
tree_dir <- path("trees", "inferred_real_trees")
tree_file <- paste0(family, ".treefile")
detlef_tree_file <- paste0(family, "_detlefsen.treefile")
tree_path <- path(tree_dir, tree_file)
detlef_tree_path <- path(tree_dir, detlef_tree_file)
tree <- read.tree(tree_path)
detlef_tree <- read.tree(detlef_tree_path)


# Create look up table seq_id -> phylum using the fasta file that has phyla labelled
phyla <- c()
fasta_lines <- readLines(labeled_seq_file)
header_lines <- str_subset(fasta_lines, "^>")
for (line in header_lines) {
    line_split <- str_split_1(line, " ")
    seq_id <- str_remove(line_split[1], "^>")
    phylum <- str_remove_all(line_split[3], "\\[|\\]")
    phyla[seq_id] <- phylum
}

# Plot tree with colored tips
# Create color mapping for phyla
colors = c(
    Acidobacteria = "#693B9E", 
    Actinobacteria = "#FBCB90", 
    Bacteroidetes =  "#E3181B",
    Chloroflexi = "#F3A19F", 
    Cyanobacteria = "#FA7F00", 
    "Deinococcus-Thermus" = "#A8CCE0", 
    Firmicutes = "#36A038", 
    Fusobacteria = "#B2E289", 
    Proteobacteria = "#147AB0"
)
tip_label_nopos = str_extract(tree$tip.label, "^[^_]*_[^_]*") 
tip_phyla = phyla[tip_label_nopos]
tip_color = ifelse(tip_phyla %in% names(colors), colors[tip_phyla], "black")
plot(tree, tip.color=tip_color, cex=0.6, type = "unrooted")
legend("topright", legend=names(colors), fill=colors, cex=0.8)


tip_label_nopos_detlef = str_split_i(detlef_tree$tip.label, "/", 1) 
tip_phyla_detlef = phyla[tip_label_nopos_detlef]
tip_color_detlef = ifelse(tip_phyla_detlef %in% names(colors), colors[tip_phyla_detlef], "black")
plot(detlef_tree, tip.color=tip_color_detlef, cex=0.6, type = "unrooted")
legend("topright", legend=names(colors), fill=colors, cex=0.8)
