"""
Simulate MSAs along phylogenetic trees using an ArDCA model fitted to a real MSA.
"""
using JLD2
using Glob
using UnPack
using ArDCA
using AncestralSequenceReconstruction
using TreeTools
using FASTX

msa_file = joinpath("msas", "real", "processed", "PF00565", "seq_msa_char.fasta")
model_file = joinpath("msas", "ardca", "pf00565_model.jld2")

# Fit ArDCA model to the PF00565 MSA
if isfile(model_file)
    @info "ArDCA model files already exists. Skipping model fitting."
    arnet, arvar = load(model_file, "arnet", "arvar")
else
    arnet, arvar = ardca(msa_file, verbose=false)
    jldsave(model_file; arnet, arvar)
end

# Check whether model has gaps in it and choose the alphabet accordingly
# Note that :ardca_aa corresponds to Alphabet("ACDEFGHIKLMNPQRSTVWY-")
alphabet = arvar.q == 21 ? :ardca_aa : ASR.Alphabet("ACDEFGHIKLMNPQRSTVWY")
ar_model = AutoRegressiveModel(arnet; alphabet=alphabet)

# Simulate along each saved tree
tree_files = glob("*/*.clean.tree", "trees/fast_trees")
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
	FASTAWriter(open(joinpath(output_dir, "$msa_id-PF00565.fa"), "w")) do writer
		for (name, seq) in all_sequences
			write(writer, FASTARecord(name, seq))
		end
	end
end