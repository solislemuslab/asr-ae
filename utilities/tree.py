from ete3 import Tree

def get_depths(tree_path, tree_format=1):
    """
    Given path to tree, return dictionary mapping internal node names to their depth, i.e. their distance to nearest leaf.
    """

    def compute_up(tree):
        """ 
        Calling this function creates an attribute called `dists_below` for every node (including leaves) in the rooted tree,
        which stores the distance from the node to all leaves below the node.
        """
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.dists_below = { node.name: 0}
            else: 
                node.dists_below = {}
                for child in node.get_children():
                    for leaf, d in child.dists_below.items():
                        node.dists_below[leaf] = d + child.dist

    def compute_down(tree):
        """
        tree must have attribute `dists_below` computed for every node
        Key idea: suppose that I'm some arbitrary node. For any leaf that's not in my subtree, the distance from me to it is 
        just the distance from my parent to it plus the distance from me to my parent. Well, if we're traversing in a preorder, then 
        the distance from my parent to it will have already been calculated. Yay.
        """
        for node in tree.traverse("preorder"):
            node.dists_all = node.dists_below.copy()
            if node.up:
                for leaf, d in node.up.dists_all.items():
                    if leaf not in node.dists_all:
                        node.dists_all[leaf] = d + node.dist


    tree = Tree(tree_path, format=tree_format)
    compute_up(tree)
    compute_down(tree)
    internal_depths = {}
    for node in tree.traverse():
        if node.is_leaf():
            continue
        internal_depths[node.name] = min(node.dists_all.values())
    return internal_depths