using Random
using Glob
using UnPack
using TreeTools
using PottsEvolver
using BioSequenceMappings

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
#############
msa_id, scaling = ARGS 
potts = read_graph("msas/potts/parameters_PF00076.dat")# Using parameters from the one and only Potts model we have for a real PFAM family
tree_file = glob("*/$msa_id.sim.trim.tree", "trees/fast_trees")
@assert length(tree_file) == 1
tree_file = tree_file[1]
num_seq = parse(Int, splitpath(tree_file)[3])
tree = read_tree(tree_file)
label_nodes!(tree)
# scale branch lengths to make reconstruction harder
s = parse(Float64, scaling)
if s != 1.0
    for node in collect(values(tree.lnodes))
        if node.isroot == false
            node.tau = s*node.tau
        end
    end
end
# How should we interpret branch lengths for simulating sequence evolution? See ?BranchLengthMeaning 
b_meaning = BranchLengthMeaning(type=:sweep, length=:round)
parameters = SamplingParameters(; Teq=0, burnin=1000, branchlength_meaning=b_meaning) # Teq doesn't mean anything when simulating along tree, but must still be specified
result = mcmc_sample(potts, tree, parameters; init=:random_aa)
@unpack leaf_sequences, internal_sequences = result
# for some reason, sampling adds weird suffixes to the names of internal nodes
internal_sequences.names = map(x -> split(x, "__")[1], internal_sequences.names)
all_sequences = cat(leaf_sequences, internal_sequences)
@assert length(all_sequences) == 2*length(tree.lleaves) - 2
write("msas/potts/raw/$num_seq/$msa_id-s$s-pottsPF00076_msa.dat", all_sequences)

