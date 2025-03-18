using Random
using LinearAlgebra
using Plots
using UnPack
using PottsEvolver
using BioSequenceMappings
using FreqTables
using TreeTools
using PrettyTables
using Statistics
# Function definitions
"""
Compute scalar contact scores for pairs of positions using Frobenius norm of q x q matrix of couplings
"""
function compute_all_contacts(potts::PottsGraph)
    L= size(potts.h, 2)
    contact_scores = zeros(L, L)
    for i in 1:L
        for j in 1:L
            if i != j
                contact_scores[i, j] = contact_scores[j, i] = norm(potts.J[:,:, i, j])
            end
        end
    end
    return contact_scores
end
"""
Compute mutual information between two columns of an MSA
"""
function compute_mi(col1::AbstractVector{T}, col2::AbstractVector{T}) where T<:Integer
    @assert length(col1) == length(col2)
    @assert all(col1 .> 0) && all(col1 .<= 21) && all(col2 .> 0) && all(col2 .<= 21)
    
    joint_counts = zeros(21, 21)

    for (a, b) in zip(col1, col2)
        joint_counts[a, b] += 1
    end

    joint_probs = joint_counts / sum(joint_counts)

    p_col1 = dropdims(sum(joint_probs, dims=2), dims=2)
    p_col2 = dropdims(sum(joint_probs, dims=1), dims=1)

    mi = 0.0
    for i in 1:21
        for j in 1:21
            if joint_probs[i, j] > 0
                mi += joint_probs[i, j] * log2(joint_probs[i, j] / (p_col1[i] * p_col2[j]))
            end
        end
    end

    return mi
end
"""
Compute mutual information between all columns of an MSA
"""
function compute_all_mis(all::Alignment)
    L = all.data.size[1]
    mi_matrix = zeros(L, L)
    for i in 1:L
        for j in 1:L
            if i != j
                mi_matrix[i, j] = mi_matrix[j, i] = compute_mi(all.data[i, :], all.data[j, :])
            end
        end
    end
    return mi_matrix
end 
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

# Read Potts model from file, subset it to a particular subsequence, and visualize contact scores 
potts = read_graph("msas/potts/parameters_PF00076.dat")
#Accentuate certain couplings
# potts.J[:,:,11,16] = -20*potts.J[:,:,11,16]
# potts.J[:,:,16,11] = transpose(potts.J[:,:,11,16])
# potts.J[:,:,14,15] = 5*potts.J[:,:,14,15]
# potts.J[:,:,15,14] = transpose(potts.J[:,:,14,15])
# potts.J[:,:,10,20] = -20*potts.J[:,:,10,20]
# potts.J[:,:,20,10] = transpose(potts.J[:,:,10,20])
# potts.J[:,:,55,56] = 5*potts.J[:,:,55,56]
# potts.J[:,:,56,55] = transpose(potts.J[:,:,55,56])
# potts.J[:,:,54,60] = 10*potts.J[:,:,54,60]
# potts.J[:,:,60,54] = transpose(potts.J[:,:,54,60])
# contact_scores = compute_all_contacts(potts)
# heatmap(contact_scores, aspect_ratio=1, color=:viridis, title="Contact scores")

# test simulation not on linear chain ########
# parameters = SamplingParameters(; Teq=10, burnin=1000) # Teq is number of steps taken between samples, burnin is number of steps taken before sampling
# M = 5000
# results = mcmc_sample(potts, M, parameters; init=:random_aa)
# all_sequences = results.sequences
# # for (seq, name) in zip(eachsequence(all_sequences), all_sequences.names)
# #     println("$name:  $(join([all_sequences.alphabet(i) for i in seq]))")
# # end
# # freqtable(all_sequences.data[10,:], all_sequences.data[20,:])
# # freqtable(all_sequences.data[14,:], all_sequences.data[15,:])
# # freqtable(all_sequences.data[11,:], all_sequences.data[16,:])
# mi_matrix = compute_all_mis(all_sequences)
# heatmap(mi_matrix, aspect_ratio=1, color=:viridis, title="Mutual Information")

# now try simulating MSA on a tree ########
tree = read_tree("trees/fast_trees/5000/COG438.sim.trim.tree")
avg_branch_lengths = mean([node.tau for node in collect(values(tree.lnodes)) if node.isroot == false])
label_nodes!(tree)
# How should we interpret branch lengths for simulating sequence evolution? See ?BranchLengthMeaning 
b_meaning = BranchLengthMeaning(type=:sweep, length=:round)
parameters = SamplingParameters(; Teq=0, burnin=1000, branchlength_meaning=b_meaning) # Teq doesn't mean anything when simulating along tree, but must still be specified
result = mcmc_sample(potts, tree, parameters; init=:random_aa)
@unpack leaf_sequences, internal_sequences = result
# for some reason, sampling adds weird suffixes to the names of internal nodes
internal_sequences.names = map(x -> split(x, "__")[1], internal_sequences.names)
all_sequences = cat(leaf_sequences, internal_sequences)
@assert length(all_sequences) == 9998
write("msas/potts/raw/5000/COG438-pottsPF00076_msa.dat", all_sequences)
#freqtable(leaf_sequences.data[55,:], leaf_sequences.data[56,:])
#mi_matrix = compute_all_mis(leaf_sequences)
#heatmap(mi_matrix, aspect_ratio=1, color=:viridis, title="Mutual Information")
# h = Highlighter(
#     (data, i, j) -> (data[i, j] > 0.24),
#     bold       = true,
#     foreground = :blue
# )
#pretty_table(mi_matrix, highlighters=h,row_labels = 1:20, crop=:none)
