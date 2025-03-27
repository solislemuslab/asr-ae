using AncestralSequenceReconstruction
using JLD2
using TreeTools
using FASTX

# Form path to data directory
data_context = joinpath("msas", "potts", "processed", "5000")
tree_family = "COG438"
potts_family = "pottsPF00076"
family = "$tree_family-$potts_family"
data_dir = joinpath(data_context, family)

# Retrieve the ardca model learned for the family
arnet = let 
    saved_model = JLD2.load(joinpath(data_dir, "ardca_model.jld2"))
    saved_model["arnet"]
end
# When wrapping the arnet model in a struct defined in the AncestralSequenceReconstruction package, 
# We specify the mapping of aa -> integer to match the mapping used in the ARDCA model (i.e. "-" comes last, not first)
ar_model = AutoRegressiveModel(arnet; alphabet=:ardca_aa) 

# Tree and fasta file 
tree_file = joinpath("trees", "fast_trees", "5000", "COG438.sim.fully_trim.tree") # make sure this tree has internal node names (assigned by the R script)
fasta_file = joinpath(data_dir, "seq_msa_char.fasta")

# ASR strategy
strategy = strategy = ASRMethod(;
    joint = false, # (default) - joint reconstruction not functional yet
    ML = true, # (default)
    verbosity = 1, # the default is 0. 
    optimize_branch_length = false, # (default: false) - optimize the branch lengths of the tree using the evolutionary model
    optimize_branch_scale = false, # (default) - optimizes the branches while keeping their relative lengths fixed. Incompatible with the previous. 
    repetitions = 1 # (default) - for Bayesian reconstruction, multiple repetitions of the reconstruction process can be done to sample likely ancestors
)

# Actual ASR
opt_tree, reconstructed_sequences = infer_ancestral(
	tree_file, fasta_file, ar_model, strategy
)	

# Write reconstructed sequences to a fasta file
ardca_dir = joinpath("reconstructions", "ardca", "potts", "5000", family)
outfasta = joinpath(ardca_dir, "reconstructed.fasta")
# make sure the output directory exists
if !isdir(ardca_dir)
    mkpath(ardca_dir)
end
begin
	FASTAWriter(open(outfasta, "w")) do writer
		for (name, seq) in reconstructed_sequences
			write(writer, FASTARecord(name, seq))
		end
	end
end