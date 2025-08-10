import sys

def reformat_potts(input_file, output_file):
    char_to_index = {char: idx for idx, char in enumerate("-ACDEFGHIKLMNPQRSTVWY")}
    reformatted_lines = []
    
    with open(input_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split()
            if parts[0] == "J":
                # Format: J int1 int2 char1 char2 val
                parts[3] = str(char_to_index[parts[3]])
                parts[4] = str(char_to_index[parts[4]])
            elif parts[0] == "h":
                # Format: h int char val
                parts[2] = str(char_to_index[parts[2]])
            reformatted_lines.append(" ".join(parts))
    
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(reformatted_lines) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reformat_potts.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file, output_file = sys.argv[1], sys.argv[2]
    reformat_potts(input_file, output_file)
