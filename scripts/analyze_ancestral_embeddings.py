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
from matplotlib.lines import Line2D
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.utils import get_directory


# Edit paths appropriately
data_path = "msas/independent/processed/10000/pevae"
tree_path = "trees/fast_trees/10000/pevae.clean.tree"
model_name = "ding_layers500_ld2_wd0.001_epoch500_2025-07-17.pt"
save = True

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
spec_leaves = []
threshes = ((-1,-1), (-1,1), (1,-1), (1,1))
for x,y in threshes:
    x_flag = (embed_leaves.iloc[:, 0] < 2.5 * x if x < 0 else embed_leaves.iloc[:, 0] > 2.5 * x) 
    y_flag = (embed_leaves.iloc[:, 1] < 2.5 * y if y < 0 else embed_leaves.iloc[:, 1] > 2.5 * y) 
    flag = x_flag*y_flag
    spec_leaf = embed_leaves[flag].index[0]
    spec_leaves.append(spec_leaf)
# Explore some specific leaves and their ancestors
# For COG28:
#spec_leaves = ['N730', 'N213', 'N255', 'N1002']
# For COG2814
#spec_leaves = ['N3752', 'N2961', 'N3489', 'N2802']
# For pevae
#spec_leaves = ['N6691', 'N5139', 'N1393', 'N6084']

spec_leaves_ancs = {}
for leaf in spec_leaves:
    spec_leaves_ancs[leaf] = []
    ancs = (t&leaf).get_ancestors()
    for anc in ancs:
        spec_leaves_ancs[leaf].append(anc.name)


## Plot 
plt.figure(figsize=(6, 4))
# Plot special leaves and their ancestors
for k in range(len(spec_leaves)):
    leaf_name = spec_leaves[k]
    data = pd.DataFrame(index = [leaf_name] + spec_leaves_ancs[leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (embed_data.loc[leaf_name, "dim0"], embed_data.loc[leaf_name, "dim1"], t.get_distance(t&leaf_name))
    num_anc = len(spec_leaves_ancs[leaf_name])
    for anc in spec_leaves_ancs[leaf_name]:
        data.loc[anc, :] = (embed_data.loc[anc, "dim0"], embed_data.loc[anc, "dim1"], t.get_distance(t&anc))

    plt.plot(data.loc[leaf_name,'mu1'], data.loc[leaf_name,'mu2'], '+r', markersize=20)
    #plt.text(data.loc[leaf_name, 'mu1'], data.loc[leaf_name, 'mu2'], leaf_name, color='red', fontsize=8, ha='center', va='center', fontweight='bold')
    plt.plot( data.loc[:, 'mu1'], data.loc[:, 'mu2'], color='gray', linestyle='-', linewidth=1.5)
    plt.scatter(data.loc[:,'mu1'], data.loc[:,'mu2'], c = data.loc[:, 'depth'], s=20, cmap=plt.get_cmap('viridis'))
    

# Plot all other embeddings in the background
plt.plot(embed_data.loc[ancestors,"dim0"], embed_data.loc[ancestors,"dim1"], 'r.', alpha = 0.1, markersize = 1, label = 'ancestral')
plt.plot(embed_data.loc[leaves,"dim0"], embed_data.loc[leaves,"dim1"], 'b.', alpha = 0.1, markersize = 1, label = 'leaf')
# Custom legend handles with higher alpha
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='red', label='ancestral',
           markersize=6, alpha=0.2),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', label='leaf',
           markersize=6, alpha=0.2)
]

plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
cbar = plt.colorbar()
cbar.set_label("distance to root")
#plt.legend(handles=legend_elements)
if save:
    model = os.path.splitext(model_name)[0]
    data = os.path.basename(data_path)
    fig_direct = os.path.join("figures", data, model)
    if not os.path.exists(fig_direct):
        os.makedirs(fig_direct)
    plt.savefig(os.path.join(fig_direct, "latent_space.pdf"), bbox_inches='tight')
else:
    plt.show()
