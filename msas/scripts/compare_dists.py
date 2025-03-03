import re
import sys
from collections import Counter
import matplotlib.pyplot as plt

# File paths
coevolution_file = 'msas/coupled/coevolution_stationary.txt'
lg_lg_file = 'msas/independent/lg_LG.PAML.txt'

# Get the FASTA file from command line arguments
if len(sys.argv) != 2:
    print("Usage: python msas/scripts/compare_dists.py <fasta_file>")
    sys.exit(1)

fasta_file = sys.argv[1]

# Determine the type of simulation based on the directory structure
if 'independent' in fasta_file:
    sim_type = 'independent'
elif 'coupled' in fasta_file or 'cherry' in fasta_file:
    sim_type = 'coupled'
else:
    raise ValueError("Unknown simulation type based on the directory structure.")

# Read sequences from fasta file
with open(fasta_file, 'r') as f:
    sequences = [line.strip() for line in f if not line.startswith('>')]

if sim_type == 'independent':
    # Extract individual amino acids
    amino_acids = ''.join(sequences)
    
    # Count the occurrences of each amino acid
    aa_counts = Counter(amino_acids)
    
    # Read stationary distribution from lg_LG.PAML.txt
    with open(lg_lg_file, 'r') as f:
        lines = f.readlines()
        stationary_distribution = list(map(float, lines[-1].strip().split()))
    
    # Normalize the counts to get observed probabilities
    total_aa = sum(aa_counts.values())
    observed_probs = {aa: count / total_aa for aa, count in aa_counts.items()}
    
    # Prepare data for plotting
    labels = "ARNDCQEGHILKMFPSTWYV" # Amino acid labels in same order as stationary distribution
    observed = [observed_probs.get(aa, 0) for aa in labels]
    expected = stationary_distribution[:len(labels)]
    xlabel = 'Expected Probability (Individual Amino Acids)'
    ylabel = 'Observed Probability (Individual Amino Acids)'

elif sim_type == 'coupled':
    # Extract all successive pairs of characters from each sequence
    pairs = []
    for seq in sequences:
        pairs.extend([seq[i:i+2] for i in range(0, len(seq) - 1, 2)])
    
    # Count the occurrences of each pair
    pair_counts = Counter(pairs)
    
    # Read coevolution probabilities
    coevolution_probs = {}
    with open(coevolution_file, 'r') as f:
        for line in f:
            if re.match(r'^[A-Z]{2}\s+\d+\.\d+', line):
                pair, prob = line.split()
                coevolution_probs[pair] = float(prob)
    
    # Normalize the counts to get probabilities
    total_pairs = sum(pair_counts.values())
    observed_probs = {pair: count / total_pairs for pair, count in pair_counts.items()}
    
    # Prepare data for plotting
    labels = sorted(set(observed_probs.keys()).union(coevolution_probs.keys()))
    observed = [observed_probs.get(pair, 0) for pair in labels]
    expected = [coevolution_probs.get(pair, 0) for pair in labels]
    xlabel = 'Expected Probability (Pairs of Amino Acids)'
    ylabel = 'Observed Probability (Pairs of Amino Acids)'

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(expected, observed)

# Add a line y=x for reference
ax.plot([0, max(expected)], [0, max(expected)], 'r--')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title('Scatter Plot of Observed vs Expected Probabilities')
ax.set_xscale('log')
ax.set_yscale('log')
# Annotate points with labels
for i, label in enumerate(labels):
    ax.annotate(label, (expected[i], observed[i]), fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()