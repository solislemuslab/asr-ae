using AncestralSequenceReconstruction
using ArDCA
using JLD2
using TreeTools
using FASTX

data_dir, tree_file = ARGS # the tree should have internal node names (assigned by the R script)
msa_file = joinpath(data_dir, "seq_msa_char.fasta") # make sure this file has exactly same sequences as leaves of the tree (also ensured by R script)

# Where will reconstructed sequences be written
ardca_dir = joinpath("reconstructions", "ardca", splitpath(data_dir)[[2,4,5]]...)
mkpath(ardca_dir)	
reconstruct_fasta = joinpath(ardca_dir, "reconstructed.fasta")
if isfile(reconstruct_fasta)
    @info "$reconstruct_fasta already exists with reconstructed sequences. ArDCA reconstruction has already been performed"
    exit()
end

# Fit ArDCA model to the sequence alignment
if isfile("$data_dir/ardca_model.jld2")
    @info "ArDCA model files already exists. Skipping model fitting."
    arnet, arvar = load("$data_dir/ardca_model.jld2", "arnet", "arvar")
else
    arnet, arvar = ardca(msa_file)
    jldsave("$data_dir/ardca_model.jld2"; arnet, arvar)
end

# Check whether model has gaps in it and choose the alphabet accordingly
# Note that :ardca_aa corresponds to Alphabet("ACDEFGHIKLMNPQRSTVWY-")
alphabet = arvar.q == 21 ? :ardca_aa : ASR.Alphabet("ACDEFGHIKLMNPQRSTVWY") 
ar_model = AutoRegressiveModel(arnet; alphabet=alphabet) 

# ASR strategy
strategy = strategy = ASRMethod(;
    joint = false, # (default) - joint reconstruction not functional yet
    ML = true, # (default)
    verbosity = 2, # the default is 0. 
    optimize_branch_length = false, # (default: false) - optimize the branch lengths of the tree using the evolutionary model
    optimize_branch_scale = false, # (default) - optimizes the branches while keeping their relative lengths fixed. Incompatible with the previous. 
    repetitions = 1 # (default) - for Bayesian reconstruction, multiple repetitions of the reconstruction process can be done to sample likely ancestors
)
# Run ASR
opt_tree, reconstructed_sequences = infer_ancestral(
	tree_file, msa_file, ar_model, strategy
)

# Write reconstructed sequences to a fasta file
FASTAWriter(open(reconstruct_fasta, "w")) do writer
	for (name, seq) in reconstructed_sequences
        # for some reason, package adds weird suffixes to the names of internal nodes
		write(writer, FASTARecord(split(name, "__")[1], seq))
	end
end
