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

#Other constants for processing the real MSAs
MAX_GAPS_IN_SEQ = 50
MAX_GAPS_IN_POS = 0.2