from networkx import Graph
import numpy as np

class Trim(Graph):
    __slots__ = {'vertices', 'faces', 'faces_of_edges'}

    def __init__(self, vertices=None, faces=None):
        Graph.__init__(self, incoming_graph_data=None)
        self.vertices = vertices
        self.faces = faces
        self.faces_of_edges = dict({})
        for i in range(0, len(faces)):
            V0 = faces[i][0]
            V1 = faces[i][1]
            V2 = faces[i][2]
            self.add_edge(V0, V1)
            self.add_edge(V1, V2)
            self.add_edge(V0, V2)

            if (V0, V1) in self.faces_of_edges:
                self.faces_of_edges[(V0, V1)].add(i)
            elif (V1, V0) in self.faces_of_edges:
                self.faces_of_edges[(V1, V0)].add(i)
            else:
                self.faces_of_edges[(V0, V1)] = set({})
                self.faces_of_edges[(V0, V1)].add(i)

            if (V0, V2) in self.faces_of_edges:
                self.faces_of_edges[(V0, V2)].add(i)
            elif (V2, V0) in self.faces_of_edges:
                self.faces_of_edges[(V2, V0)].add(i)
            else:
                self.faces_of_edges[(V0, V2)] = set({})
                self.faces_of_edges[(V0, V2)].add(i)

            if (V1, V2) in self.faces_of_edges:
                self.faces_of_edges[(V1, V2)].add(i)
            elif (V2, V1) in self.faces_of_edges:
                self.faces_of_edges[(V2, V1)].add(i)
            else:
                self.faces_of_edges[(V1, V2)] = set({})
                self.faces_of_edges[(V1, V2)].add(i)

    def is_boundry_edge(self, edge):
        if edge not in self.edges:
            print('Is Boundry Edge: Not a Edge!')
            return None
        else:
            if (edge[0], edge[1]) in self.faces_of_edges.keys():
                faces = self.faces_of_edges[edge]
            else:
                faces = self.faces_of_edges[(edge[1], edge[0])]
            if len(faces) <= 1:
                return True
            else:
                return False

    def is_boundry_vertex(self, vertex):
        if vertex >= len(self.nodes):
            return None
        else:
            neighbour = self.adj[vertex]
            for V in neighbour:
                if (V, vertex) in self.faces_of_edges.keys():
                    if self.is_boundry_edge(edge=(V, vertex)):
                        return True
                    else:
                        pass
                elif (vertex, V) in self.faces_of_edges.keys():
                    if self.is_boundry_edge(edge=(V, vertex)):
                        return True
                    else:
                        pass
                else:
                    return None

            return False

if __name__ == '__main__':
    G = Trim(vertices=['a', 'b', 'c', 'd', 'e'], faces=[(0, 1, 4), (1, 4, 2), (2, 4, 3), (0, 4, 3)])
    if (0, 1) in G.edges:
        print('AAA')
    if (1, 0) in G.edges:
        print('AAA')
    a = G.is_boundry_vertex(3)
    x = 1
