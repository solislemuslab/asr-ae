def read_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # First 20 lines contain the exchange matrix
    exchange_lines = lines[:20]
    
    # Convert to a list of floats
    mat_exchanges = []
    for line in exchange_lines:
        row = list(map(float, line.split()))
        mat_exchanges.append(row)
    
    # 22nd line contains the frequencies
    freqs = list(map(float, lines[21].split()))

    return mat_exchanges, freqs

def flatten_lower_triangular_column_major(matrix):
    n = len(matrix)
    flattened = []
    
    for col in range(n):
        for row in range(col + 1, n):
            flattened.append(matrix[row][col])
    
    return flattened

def main():
    file_path = './independent_sims/lg_LG.PAML.txt'
    mat_exchanges, freqs = read_from_file(file_path)
    flat_exchanges = flatten_lower_triangular_column_major(mat_exchanges)
    flat_exchanges = ','.join(map(str, flat_exchanges))
    freqs = ','.join(map(str, freqs))
    print(flat_exchanges + " " + freqs)

if __name__ == "__main__":
    main()
