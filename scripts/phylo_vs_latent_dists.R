suppressPackageStartupMessages({
    library(fs)
    library(ape)
    library(tidyverse)
})

#======= Edit the following code to run the script for the desired family/model ==========#
manuscript_figure <- TRUE
if (manuscript_figure) {
    runs <- tribble(
        ~data_path,                                      ~model,
        "msas/independent/processed/1250/COG28-l100-s1-a0.5",   "ding-alldata_layers500_ld2_wd0.005_epoch500_2026-06-08",
        "msas/independent/processed/1250/COG28-l100-s1-a0.5",   "ding-alldata_layers500_ld20_wd0.01_epoch500_2026-06-08",
        "msas/independent/processed/5000/COG2814-l100-s1-a0.5", "ding-alldata_layers500_ld2_wd0.001_epoch500_2026-06-08",
        "msas/independent/processed/5000/COG2814-l100-s1-a0.5", "ding-alldata_layers500_ld20_wd0.005_epoch500_2026-06-08",
        "msas/independent/processed/10000/pevae",               "ding-alldata_layers500_ld2_wd0.001_epoch500_2026-06-08",
        "msas/independent/processed/10000/pevae",               "ding-alldata_layers500_ld20_wd0.005_epoch500_2026-06-08",
    )
} else {
    data_path <- commandArgs(trailingOnly = TRUE)[1]
    model <- commandArgs(trailingOnly = TRUE)[2]
    if (str_detect(model, "\\.pt$")) model <- str_remove(model, "\\.pt$")
    runs <- tibble(data_path = data_path, model = model)
}
#==========================================================================================#

compute_pairwise_dists <- function(data_path, model) {
    msa_id <- path_file(data_path)
    family <- str_extract(msa_id, "^[a-zA-Z0-9]+")
    sim <- !str_detect(data_path, "real")
    if (sim) {
        sim_type <- path_split(data_path)[[1]][2]
        n_seq <- as.integer(path_file(path_dir(data_path)))
    }

    # Read tree
    if (sim) {
        tree_dir <- if (family == "pevae") path("trees", n_seq) else path("trees", "fast_trees", n_seq)
        tree_file <- paste0(family, ".clean.tree")
    } else {
        tree_dir <- path("trees", "inferred_real_trees")
        tree_file <- paste0(family, "_cleaned.treefile")
    }
    tree <- read.tree(path(tree_dir, tree_file))

    # Read leaf embeddings
    embed_dir <- path("embeddings", "data")
    if (sim) {
        embed_path <- path(embed_dir, sim_type, n_seq, msa_id, paste0(model, "_embeddings.csv"))
    } else {
        embed_path <- path(embed_dir, "real", msa_id, paste0(model, "_embeddings.csv"))
    }
    embeds <- read_csv(embed_path, show_col_types = FALSE)

    if (sim) {
        embed_leaves <- embeds |> filter(str_detect(id, "^N"))
    } else {
        embed_leaves <- embeds
    }

    # Sample leaves
    set.seed(42)
    n_sample <- min(500, nrow(embed_leaves))
    sampled <- embed_leaves |> slice_sample(n = n_sample)
    sampled_ids <- sampled$id

    # Phylogenetic (cophenetic) distances
    phylo_dists <- cophenetic.phylo(tree)
    phylo_dists <- phylo_dists[sampled_ids, sampled_ids]

    # Euclidean distances in embedding space
    dim_cols <- str_subset(colnames(sampled), "^dim")
    embed_mat <- sampled |> select(all_of(dim_cols)) |> as.matrix()
    rownames(embed_mat) <- sampled_ids
    latent_dists <- as.matrix(dist(embed_mat))

    # Extract upper triangle
    idx <- which(upper.tri(phylo_dists), arr.ind = TRUE)
    tibble(
        phylo_dist = phylo_dists[idx],
        latent_dist = latent_dists[idx]
    )
}

# Compute pairwise distances for each run
all_pairs <- pmap_dfr(runs, function(data_path, model) {
    family <- str_extract(path_file(data_path), "^[a-zA-Z0-9]+")
    ld <- str_extract(model, "ld[0-9]+")
    cat(sprintf("Processing %s, %s ...\n", family, ld))
    df <- compute_pairwise_dists(data_path, model)
    df$family <- family
    df$ld <- ld
    df
})

if (manuscript_figure) {
    # Compute per-panel correlation labels
    cor_labels <- all_pairs |>
        group_by(family, ld) |>
        summarise(
            pearson = cor(phylo_dist, latent_dist, method = "pearson"),
            .groups = "drop"
        ) |>
        mutate(label = sprintf("r = %.2f", pearson))

    family_labels <- c(COG28 = "1250 tips", COG2814 = "5000 tips", pevae = "10000 tips")
    ld_labels <- c(ld2 = "2-dimensional VAE", ld20 = "20-dimensional VAE")
    all_pairs$family <- factor(all_pairs$family, levels = names(family_labels), labels = family_labels)
    all_pairs$ld <- factor(all_pairs$ld, levels = names(ld_labels), labels = ld_labels)
    cor_labels$family <- factor(cor_labels$family, levels = names(family_labels), labels = family_labels)
    cor_labels$ld <- factor(cor_labels$ld, levels = names(ld_labels), labels = ld_labels)

    p <- ggplot(all_pairs, aes(x = latent_dist, y = phylo_dist)) +
        geom_point(alpha = 0.05, size = 0.3) +
        geom_smooth(method = "lm", color = "red", linewidth = 0.8) +
        geom_text(
            data = cor_labels,
            aes(x = Inf, y = Inf, label = label),
            hjust = 1.1, vjust = 1.3, size = 4, inherit.aes = FALSE
        ) +
        facet_grid(family ~ ld, scales = "free") +
        labs(
            x = "Euclidean distance in latent space",
            y = "Phylogenetic distance (cophenetic)"
        ) +
        theme_minimal(base_size = 14) +
        theme(strip.text = element_text(size = 13))

    plot_dir <- "figures"
    if (!dir.exists(plot_dir)) dir_create(plot_dir, recurse = TRUE)
    ggsave(
        "ind_phylo-vs-latent-dists.png",
        plot = p,
        path = plot_dir,
        width = 8,
        height = 9,
        bg = "white"
    )
    cat("Plot saved to", path(plot_dir, "ind_phylo-vs-latent-dists.png"), "\n")
} else {
    msa_id <- path_file(runs$data_path)
    family <- str_extract(msa_id, "^[a-zA-Z0-9]+")
    sim <- !str_detect(runs$data_path, "real")
    model <- runs$model

    plot_dir <- if (sim) {
        n_seq <- as.integer(path_file(path_dir(runs$data_path)))
        sim_type <- path_split(runs$data_path)[[1]][2]
        path("plots", sim_type, n_seq, msa_id)
    } else {
        path("plots", "real", msa_id)
    }
    if (!dir.exists(plot_dir)) dir_create(plot_dir, recurse = TRUE)

    pearson <- cor(all_pairs$phylo_dist, all_pairs$latent_dist, method = "pearson")
    spearman <- cor(all_pairs$phylo_dist, all_pairs$latent_dist, method = "spearman")
    cat(sprintf("Pairs: %d  Pearson: %.3f  Spearman: %.3f\n", nrow(all_pairs), pearson, spearman))

    p <- ggplot(all_pairs, aes(x = latent_dist, y = phylo_dist)) +
        geom_point(alpha = 0.05, size = 0.3) +
        geom_smooth(method = "lm", color = "red", linewidth = 0.8) +
        annotate("text",
            x = Inf, y = Inf,
            hjust = 1.1, vjust = 1.5, size = 5,
            label = sprintf("Pearson = %.3f\nSpearman = %.3f", pearson, spearman)
        ) +
        labs(
            x = "Euclidean distance in latent space",
            y = "Phylogenetic distance (cophenetic)"
        ) +
        theme_minimal(base_size = 16)

    ggsave(
        paste0(model, "_phylo_vs_latent_dists.png"),
        plot = p,
        path = plot_dir,
        width = 7, height = 6, bg = "white"
    )
    cat("Plot saved to", path(plot_dir, paste0(model, "_phylo_vs_latent_dists.png")), "\n")
}
