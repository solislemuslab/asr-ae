# Amino acid orders for integer encoding the real and simulated MSAs
AA = [ 
    "A", "R", "N", "D", "C", 
    "Q", "E", "G", "H", "I", 
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
REAL_AA = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

# Unknown symols
UNKNOWN = ['-', '.', 'X', 'B', 'Z', 'J'] 
# For the real MSAs, we chose from the set UNKNOWN the symbol '.' to represent all unknowns in the processed character MSA of leaf sequences
# In the simulated MSAs, we chose from the set UNKNOWN the symbol '-' to represent all unknowns in the processed character MSA of leaf sequences

#Other constants for processing the real MSAs (fraction of positions or sequences that are gaps to justify removal, respectively)
MAX_GAPS_IN_SEQ = 0.2
MAX_GAPS_IN_POS = 0.2

# Sequence from an Archaea that is used for outgroup rooting for the pf00565 family
outgroup_pf00565_id = "A0A060HE43_9ARCH/86-213"
