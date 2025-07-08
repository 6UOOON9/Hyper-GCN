import sys

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, hyper_joints=0, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.hyper_joints = hyper_joints
        self.A = self.get_adjacency_matrix(labeling_mode)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_ensemble':
            A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, 8)
        elif labeling_mode == 'virtual_ensemble':
            A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.hyper_joints, 8)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    g = Graph(labeling_mode='virtual_spatial')
    print()