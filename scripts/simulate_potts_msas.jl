#=
simulate_potts_msas.jl

Simulates evolution by Markov sampling from a fitted Potts model along phylogenetic trees.
Output MSAs are saved inside msas/potts/raw.

Usage:
\$ julia --project=. scripts/simulate_potts_msas.jl --family pf00565
\$ julia --project=. scripts/simulate_potts_msas.jl --family pf00072 --scale 1.0 --tree trees/fast_trees/1250/some.clean.tree
=#

using ArgParse
using Random
using Glob
using UnPack
using TreeTools
using PottsEvolver
using BioSequenceMappings

###### Argument parsing ######
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--family", "-f"
            arg_type = String
            required = true
            help = "Identifier (e.g. pf00565, pf00072) of protein family to which Potts model was fit. Parameters must exist at msas/potts/<family>_params_intindex.dat"
        "--scale", "-s"
            arg_type = Float64
            default = nothing
            help = "Branch length scaling factor. If not specified, simulates at both 1.0 and 2.0."
        "--tree", "-t"
            arg_type = String
            default = nothing
            help = "Specific tree file to simulate on. If not specified, uses all trees in trees/fast_trees."
    end
    return parse_args(s)
end

args = parse_commandline()

family = args["family"]
params_file = "msas/potts/$(family)_params_intindex.dat"
if !isfile(params_file)
    error("Parameters file not found: $params_file")
end

scales = isnothing(args["scale"]) ? [1.0, 2.0] : [args["scale"]]

if !isnothing(args["tree"])
    if !isfile(args["tree"])
        error("Specified tree file does not exist: $(args["tree"])")
    end
    tree_files = [args["tree"]]
else
    tree_files = glob("*/*.clean.tree", "trees/fast_trees")
end

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

family_upper = uppercase(family)

############### Main script ########################
potts = read_graph(params_file)

for s in scales
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
        @assert length(all_sequences) == length(tree.lnodes) 
        # Write the MSA to file
        output_dir = "msas/potts/raw/$num_seq"
        mkpath(output_dir)
        write(joinpath(output_dir, "$msa_id-s$s-potts$family_upper.fa"), all_sequences)
    end
end
