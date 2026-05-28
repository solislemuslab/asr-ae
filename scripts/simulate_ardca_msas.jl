#=
simulate_ardca_msas.jl

Fits an autoregressive model (ArDCA) to a pre-processed MSA and simulates evolution
along phylogenetic trees.

Output MSAs are saved inside msas/ardca/raw.

Usage:
\$ julia --project=. scripts/simulate_ardca_msas.jl --family PF00565
\$ julia --project=. scripts/simulate_ardca_msas.jl --family PF00072 --tree trees/fast_trees/1250/some.clean.tree
=#

using ArgParse
using JLD2
using Glob
using UnPack
using ArDCA
using AncestralSequenceReconstruction
using TreeTools
using FASTX

###### Argument parsing ######
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--family", "-f"
            arg_type = String
            required = true
            help = "Protein family identifier (e.g. PF00565, PF00072). Pre-processed MSA must exist at msas/real/processed/<family>/seq_msa_char.fasta"
        "--tree", "-t"
            arg_type = String
            default = nothing
            help = "Specific tree file to simulate on. If not specified, uses all trees in trees/fast_trees."
    end
    return parse_args(s)
end

args = parse_commandline()

family = args["family"]
family_lower = lowercase(family)

msa_file = joinpath("msas", "real", "processed", family, "seq_msa_char.fasta")
if !isfile(msa_file)
    error("MSA file not found: $msa_file")
end
model_file = joinpath("msas", "ardca", "$(family_lower)_model.jld2")

# Fit ArDCA model to the MSA
if isfile(model_file)
    @info "ArDCA model file already exists at $model_file. Skipping model fitting."
    arnet, arvar = load(model_file, "arnet", "arvar")
else
    arnet, arvar = ardca(msa_file, verbose=false)
    jldsave(model_file; arnet, arvar)
end

# Check whether model has gaps in it and choose the alphabet accordingly
# Note that :ardca_aa corresponds to Alphabet("ACDEFGHIKLMNPQRSTVWY-")
alphabet = arvar.q == 21 ? :ardca_aa : ASR.Alphabet("ACDEFGHIKLMNPQRSTVWY")
ar_model = AutoRegressiveModel(arnet; alphabet=alphabet)

# Simulate along trees
if !isnothing(args["tree"])
    if !isfile(args["tree"])
        error("Specified tree file does not exist: $(args["tree"])")
    end
    tree_files = [args["tree"]]
else
    tree_files = glob("*/*.clean.tree", "trees/fast_trees")
end

for tree_file in tree_files
    tree = read_tree(tree_file)
    println("Simulating along tree $(basename(tree_file))... ")
    # Simulate evolution
    @unpack leaf_sequences, internal_sequences, tree = ASR.Simulate.evolve(tree, ar_model)
    all_sequences = merge(leaf_sequences, internal_sequences)
    @assert length(all_sequences) == length(tree.lnodes)
    # Save
    num_seq = parse(Int, splitpath(tree_file)[3])
    msa_id = split(basename(tree_file), ".")[1]
    output_dir = joinpath("msas", "ardca", "raw", string(num_seq))
    mkpath(output_dir)
	FASTAWriter(open(joinpath(output_dir, "$msa_id-$family.fa"), "w")) do writer
		for (name, seq) in all_sequences
			write(writer, FASTARecord(name, seq))
		end
	end
end