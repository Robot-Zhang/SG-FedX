import numpy as np
from treelib import Tree


class CommunityTree:
    def __init__(self, clients, hiera_part=None):
        """hiera_part: [(int-num_par, ndarray-part) ... ], low->high"""
        self.tree = Tree()
        # add server as root
        self.tree.create_node('server', 'server')
        for c in clients:
            self.tree.create_node(tag=c.id, identifier=c.id, parent='server')

        self.client_leaves = [self.tree.get_node(i) for i in range(len(clients))]
        self.new_id = len(clients)
        if hiera_part is not None: self._init_by_hiera_part(hiera_part)

        self.hiera_part = hiera_part

    def insert_com(self, children: [int], parent='server'):
        """insert community into tree, return the community id"""
        self.tree.create_node(tag=self.new_id, identifier=self.new_id, parent=parent)
        for child in children:
            self.tree.move_node(child, self.new_id)
        self.new_id += 1
        return self.new_id - 1

    def _init_by_hiera_part(self, hiera_part):
        """init tree"""
        # level 0-1
        level, com_leaves = 0, []
        for num, part in hiera_part:
            if num > 1:  # when num=1, means the root
                new_com_leaves = []
                for i in range(num):  # regard each part as a community
                    client_ids = list(np.argwhere(part == i).reshape(-1))
                    if level == 0:  # if level 0, insert com with leaves as child
                        com_id = self.insert_com(client_ids)
                    else:  # else, check insert com with other com as child
                        # a. get low com with same client leaves
                        high_set, children = set(client_ids), []
                        for com_id, low_com_client_ids in com_leaves:
                            low_set = set(low_com_client_ids)
                            if (low_set & high_set) == low_set: children.append(com_id)
                        # b. insert the low com
                        com_id = self.insert_com(children)
                    new_com_leaves.append((com_id, client_ids))
                com_leaves = new_com_leaves
                level += 1

    def rise(self, node):
        # TODO delete
        old_parent = self.tree.parent(node.identifier)
        new_parent = self.tree.parent(old_parent.identifier)
        self.tree.move_node(node.identifier, new_parent.identifier)
        self.remove_redundant()

    def remove_redundant(self):
        # TODO delete
        """remove redundant communities with only one leaf (or none leaves)"""
        count = 1
        while count > 0:
            count = 0
            for nid in self.tree.expand_tree(filter=lambda x: x not in self.client_leaves):
                if len(self.tree.leaves(nid)) <= 1:
                    parent = self.tree.parent(nid)
                    for child in self.tree.children(nid):
                        self.tree.move_node(child.identifier, parent.identifier)
                    self.tree.remove_node(nid)
                    count += 1
                    break

    def get_alpha_nodes(self):
        """iterator alpha (non-root) node in the tree"""
        alpha_ids = list(self.tree.expand_tree())
        alpha_ids.remove('server')
        return [self.tree[nid] for nid in alpha_ids]

    def get_children(self, node):
        return self.tree.children(node.identifier)

    def get_parent(self, node):
        return self.tree.parent(node.identifier)

    def get_leaves(self, node):
        leaves = []
        for n in self.tree.leaves(node.identifier):
            if n in self.client_leaves:
                leaves.append(n)
        return leaves

    def get_communities(self, depth=1):
        """return clients' id by community of given depth"""
        ttt = self.tree.depth()
        assert depth < self.tree.depth(), \
            f"community tree depth is only {self.tree.depth()}, {depth} too deep."

        # 1. get community node
        com_nodes = self.tree.children('server')
        for _ in range(depth - 1):
            children = []
            for node in com_nodes:
                children += self.tree.children(node.identifier)
            com_nodes = children

        # 2. get clients by community node
        community_leaves = [self.get_leaves(node) for node in com_nodes]
        communities = [[leaf.identifier for leaf in com] for com in community_leaves]
        return communities

    def show(self):
        print(self.tree)
