import os 
import sys
from ete3 import Tree
import pandas as pd
import matplotlib as mpl
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.utils import get_directory


# Edit paths appropriately
data_path = "msas/independent/processed/5000/COG438-l150-s1-a0.5"
tree_path = "trees/fast_trees/5000/COG438.clean.tree"
model_name = "ding_layers500_ld2_wd0.001_epoch500_2025-07-17.pt"

# read embeddings
embeddings_name = os.path.splitext(model_name)[0] + "_embeddings.csv"
embeddings_dir = get_directory(data_path, "embeddings", data_subfolder=True)
embed_data = pd.read_csv(f"{embeddings_dir}/{embeddings_name}")
embed_data = embed_data.set_index("id")
leaves = []
ancestors = []
for id in embed_data.index:
    if id[0] == "N":
        leaves.append(id)
    else:
        ancestors.append(id)

# read tree
t = Tree(tree_path, format = 1)


embed_leaves = embed_data.loc[leaves, :]
# leaf_key = np.array(key)[leaf_idx]
# flag = (leaf_mu[:, 0] < -4) * (leaf_mu[:,1] > 2) * (leaf_mu[:,1] < 4)
# leaf_mu = leaf_mu[flag]
# leaf_key = leaf_key[flag]
# idx = leaf_key[np.argmin(leaf_mu[:,0])]

# Explore some specific leaves and their ancestors
spec_leaves = ['N730', 'N213', 'N255', 'N1002']
#spec_leaves = ['N10', 'N50', 'N100', 'N290']
spec_leaves_ancs = {}
for leaf in spec_leaves:
    spec_leaves_ancs[leaf] = []
    ancs = (t&leaf).get_ancestors()
    for anc in ancs:
        spec_leaves_ancs[leaf].append(anc.name)


## Plot 
fig = plt.figure(1)
fig.clf()
# Plot special leaves and their ancestors
for k in range(len(spec_leaves)):
    leaf_name = spec_leaves[k]
    data = pd.DataFrame(index = [leaf_name] + spec_leaves_ancs[leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (embed_data.loc[leaf_name, "dim0"], embed_data.loc[leaf_name, "dim1"], t.get_distance(t&leaf_name))
    num_anc = len(spec_leaves_ancs[leaf_name])
    for anc in spec_leaves_ancs[leaf_name]:
        data.loc[anc, :] = (embed_data.loc[anc, "dim0"], embed_data.loc[anc, "dim1"], t.get_distance(t&anc))

    plt.scatter(data.loc[:,'mu1'], data.loc[:,'mu2'], c = data.loc[:, 'depth'], cmap = plt.get_cmap('viridis'))
    plt.plot( data.loc[:, 'mu1'], data.loc[:, 'mu2'], color='gray', linestyle='-', linewidth=1.5)
    plt.text(data.loc[leaf_name, 'mu1'], data.loc[leaf_name, 'mu2'], leaf_name, color='red', fontsize=8, ha='center', va='center', fontweight='bold')

# Plot all other embeddings in the background
plt.plot(embed_data.loc[ancestors,"dim0"], embed_data.loc[ancestors,"dim1"], 'r.', alpha = 0.1, markersize = 2, label = 'ancestral')
plt.plot(embed_data.loc[leaves,"dim0"], embed_data.loc[leaves,"dim1"], 'b.', alpha = 0.1, markersize = 2, label = 'leaf')

plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.colorbar()
plt.legend(markerscale = 2)
plt.tight_layout()
plt.show()
