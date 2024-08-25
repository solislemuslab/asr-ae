def to_fasta(f_in, f_out, keep = None):
    if keep:
        def process_line(line):
            id, seq = line.split()
            if id in keep:
                return f">{id}\n{seq}\n"
            else: 
                return ""
    else:
        def process_line(line):
            id, seq = line.split()
            return f">{id}\n{seq}\n"
    with open(f_in) as in_file, open(f_out, "w") as out_file:
        for line in in_file:
            processed_line = process_line(line)
            out_file.write(processed_line)
