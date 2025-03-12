import argparse
from itertools import combinations
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compute_mutual_information(col1, col2):
    joint_counts = np.zeros((20, 20))
    for a, b in zip(col1, col2):
        joint_counts[a-1, b-1] += 1 #subtract 1 to convert from 1-indexing to 0-indexing

    joint_probs = joint_counts / joint_counts.sum()

    p_col1 = joint_probs.sum(axis=1)
    p_col2 = joint_probs.sum(axis=0)

    mi = 0.0
    for i in range(20):
        for j in range(20):
            if joint_probs[i, j] > 0:
                mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (p_col1[i] * p_col2[j]))
    return mi


def plot_mi_heatmap(msa, output_dir):

    num_sites = msa.shape[1]
    mi_matrix = np.zeros((num_sites, num_sites))

    for i, j in combinations(range(num_sites), 2):
        mi = compute_mutual_information(msa[:, i], msa[:, j])
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

    plt.figure(figsize=(10, 8))
    plt.imshow(mi_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Mutual Information")
    plt.title("Mutual Information Between Sites")
    plt.xlabel("Site")
    plt.ylabel("Site")
    plt.savefig(output_dir + "/mi_heatmap.png")


def main(args):
    with open(args.msa, "rb") as file:
        msa_int = pickle.load(file)
    plot_mi_heatmap(msa_int, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msa",
        type=str,
        help="/path/ to the pickle file that contains the integer encoded msa",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="/path/ to output directory where the plot will be saved",
    )
    args = parser.parse_args()

    main(args)