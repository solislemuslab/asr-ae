using ArDCA
using JLD2
# The mapping between the aminoacid symbols and the integers uses this table:
#A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
#1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# gaps represented with 21 (instead of 0)


# msa_int = ArDCA.read_fasta_alignment("msas/potts/processed/5000/COG438-pottsPF00076/seq_msa_char.fasta", 1)
# msa_int, idx_og = ArDCA.remove_duplicate_sequences(msa_int)
# W, Meff = ArDCA.compute_weights(msa_int, :auto)
# W ./= sum(W)
tree_family = "COG438"
potts_family = "pottsPF00076"
family = "$tree_family-$potts_family"
data_dir = joinpath("msas/potts/processed/5000", family)
arnet, arvar = ardca("$data_dir/seq_msa_char.fasta")
jldsave("$data_dir/ardca_model.jld2"; arnet, arvar)
