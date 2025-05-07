using Random
using Glob
using UnPack
using TreeTools
using PottsEvolver
using BioSequenceMappings
###### Function definitions ######
"""
Label nodes of a tree according to a preorder traversal
"""
function label_nodes!(tree)
    counter = length(tree.lleaves) + 1
    for node in traversal(tree, :preorder, leaves=false)
        label!(tree, node, counter)
        counter += 1
    end
end
############### Main script ########################
# Factor by which we scale the branch lengths of the trees
scale=[1., 2.]
# Get the potts model object
potts = read_graph("msas/potts/pf000565_params.dat")
# Get tree files that we will simulate MSAs along
tree_files = glob("*/*.clean.tree", "trees/fast_trees")
@assert length(tree_files) == 14
# Iterate over trees
for s in scale
    for tree_file in tree_files
        num_seq = parse(Int, splitpath(tree_file)[3])
        msa_id = split(basename(tree_file), ".")[1]
        tree = read_tree(tree_file)
        # scale branch lengths of tree
        if s != 1.0
            for node in collect(values(tree.lnodes))
                if node.isroot == false
                    node.tau = s*node.tau
                end
            end
        end
        # How should we interpret branch lengths for simulating sequence evolution? See ?BranchLengthMeaning 
        b_meaning = BranchLengthMeaning(type=:sweep, length=:round)
        # Set MCMC configurations 
        parameters = SamplingParameters(; Teq=0, burnin=1000, branchlength_meaning=b_meaning) # Teq doesn't mean anything when simulating along tree, but must still be specified
        # Simulate the MSA with MCMC sampling according to the Potts model along the tree
        result = mcmc_sample(potts, tree, parameters; init=:random_aa)
        @unpack leaf_sequences, internal_sequences = result
        # for some reason, sampling adds weird suffixes to the names of internal nodes
        #internal_sequences.names = map(x -> split(x, "__")[1], internal_sequences.names)
        all_sequences = cat(leaf_sequences, internal_sequences)
        @assert length(all_sequences) in [2*length(tree.lleaves) - 1, 2*length(tree.lleaves) - 2] 
        # Write the MSA to file
        output_dir = "msas/potts/raw/$num_seq"
        mkpath(output_dir)
        write("$output_dir/$msa_id-s$s-pottsPF00565.fa", all_sequences)
    end
end
