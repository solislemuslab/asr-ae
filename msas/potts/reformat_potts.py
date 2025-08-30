import sys

def reformat_potts(input_file, output_file, reverse=False):
    chars = "-ACDEFGHIKLMNPQRSTVWY"
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char in enumerate(chars)}
    
    reformatted_lines = []
    
    with open(input_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split()
            if not parts:
                continue
            if not reverse:
                if parts[0] == "J":
                    # Format: J int1 int2 char1 char2 val
                    parts[3] = str(char_to_index[parts[3]])
                    parts[4] = str(char_to_index[parts[4]])
                elif parts[0] == "h":
                    # Format: h int char val
                    parts[2] = str(char_to_index[parts[2]])
            else:
                if parts[0] == "J":
                    parts[3] = index_to_char[int(parts[3])]
                    parts[4] = index_to_char[int(parts[4])]
                elif parts[0] == "h":
                    parts[2] = index_to_char[int(parts[2])]
            reformatted_lines.append(" ".join(parts))
    
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(reformatted_lines) + "\n")

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python reformat_potts.py <input_file> <output_file> [--reverse]")
        sys.exit(1)
    
    input_file, output_file = sys.argv[1], sys.argv[2]
    reverse = len(sys.argv) == 4 and sys.argv[3] == "--reverse"
    reformat_potts(input_file, output_file, reverse)