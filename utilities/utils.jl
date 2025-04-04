using PottsEvolver
using BioSequenceMappings
using LinearAlgebra

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