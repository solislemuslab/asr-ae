import os

def get_directory(data_path, MSA_id, folder, data_subfolder = False):
    """
    Take data path and MSA id and return the directory where the data is stored.
    """
    if MSA_id[0:3] == "COG": # this is a simulated dataset
        sim_type = os.path.dirname(data_path).split("/")[1] #either coupled or independent
        num_seqs = os.path.dirname(data_path).split("/")[-1] 
        dir =  f"{folder}/{sim_type}/{num_seqs}/{MSA_id}"
    else:
        dir = f"{folder}/real/{MSA_id}"
    if data_subfolder:
        dir_list = dir.split("/")
        dir_list.insert(1, "data")
        dir = ("/").join(dir_list)
    return dir

