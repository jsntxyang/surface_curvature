from visualize import plot_surface2
from skimage import measure
import numpy as np
from Data_Treatment import read_data, read_npy_data, grid_interpolation, int_transfer_coordinate, unit_grid, save_data
from Surface_Cut import remove_surface
from Tri_Mesh import Trim
from trimesh.curvature import vertex_defects
from matplotlib import cm
from Constrain_Functions import in_bz

class fermicurve:
    __slots__ = {'Kx', 'Ky', 'Kz', 'Ek', 'Vk',
                 '__triangle_mesh', '__faces_defect',
                 'data',
                 'mesh_parameter',
                 'plot_parameter',
                 }

    def __init__(self, file=None, Kx=None, Ky=None, Kz=None, Ek=None, Vk=None, mesh_parameter=None,
                 plot_parameter=None):
        self.mesh_parameter = {'N': 60, 'N_average': 10, 'isoenergy': 0.0, 'a': 4.07, 'scale': 1.0}
        self.plot_parameter = {'clip_min': 0, 'clip_max': 1e-4, 'N_cmap': 50}

        if file:
            if any([Kx, Ky, Kz, Ek, Vk]):
                raise ValueError('Kx, Ky, Kz, Ek, Vk parameters are ignored when file parameter is provided.')
            else:
                file_type = file.split('.')[-1]
                if file_type == 'npy':
                    self.Kx, self.Ky, self.Kz, self.Ek, self.data = read_npy_data(file=file)
                elif file_type == 'frmsf':
                    self.Kx, self.Ky, self.Kz, self.Ek, self.data = read_data(file=file)
                    self.Ek = self.Ek[0, :, :, :]

                else:
                    raise TypeError('%s type file cannot be opened.' % file_type)
        else:
            if not all([Kx, Ky, Kz, Ek, Vk]):
                raise ValueError('Kx, Ky, Kz, Ek, Vk parameters must provided when the file parameter is not provided.')
            else:
                self.Kx = Kx
                self.Ky = Ky
                self.Kz = Kz
                self.Ek = Ek
                self.Vk = Vk

        for u in mesh_parameter.keys():
            if u not in self.mesh_parameter.keys():
                raise ValueError('mesh_parameter does not accept parameter %s', u)
            else:
                self.mesh_parameter[u] = mesh_parameter[u]

        for u in plot_parameter.keys():
            if u not in self.plot_parameter.keys():
                raise ValueError('plot_parameter does not accept parameter %s', u)
            else:
                self.plot_parameter[u] = plot_parameter[u]

    def create_mash(self):
        N = self.mesh_parameter['N']
        isoenergy = self.mesh_parameter['isoenergy']
        N_average = self.mesh_parameter['N_average']

        #origional_data = int_transfer_coordinate(self.Ek[0, :, :, :])
        origional_data = int_transfer_coordinate(self.Ek)
        fun = grid_interpolation(origional_data)
        coordinate = unit_grid(N, N, N)
        data = fun(coordinate)

        NewK_x = (self.Ky + self.Kz)
        NewK_y = (self.Kx + self.Kz)
        NewK_z = (self.Kx + self.Ky)
        vertices, faces, normals, values = measure.marching_cubes(data, isoenergy)

        NewK = np.array([NewK_x, NewK_y, NewK_z])
        #vertices = vertices @ NewK / N - (NewK_x + NewK_y + NewK_z) / 2.0 - (NewK_x + NewK_y + NewK_z) / 50
        vertices = vertices @ NewK / N - (NewK_x + NewK_y + NewK_z) / 2.0
        TM = Trim(vertices=vertices, faces=faces, vertex_normals=normals)
        V_defect = vertex_defects(TM)

        neigh = TM.vertex_neighbors
        for i in range(0, N_average):
            A = np.array([sum(V_defect[neigh[i]]) / len(neigh[i]) for i in range(0, len(V_defect))])
            V_defect = A

        F_defect = V_defect[TM.faces].sum(axis=1) / 3
        faces = TM.faces
        vertices = TM.vertices
        normals = TM.vertex_normals
        del TM

        face_del_list = remove_surface(vertices,
                                       faces,
                                       in_bz,
                                       a=self.mesh_parameter['a'],
                                       scale=self.mesh_parameter['scale'],
                                       )
        F_defect = np.delete(F_defect, face_del_list, 0)
        faces = np.delete(faces, face_del_list, 0)
        self.__triangle_mesh = Trim(vertices=vertices, faces=faces, vertex_normals=normals)
        self.__faces_defect = F_defect
        del faces, vertices

    def save_mesh(self, name=None):
        N = self.mesh_parameter['N']
        isoenergy = self.mesh_parameter['isoenergy']
        if name:
            save_data(file=name, vertices=self.__triangle_mesh.vertices,
                      vertices_normal=self.__triangle_mesh.vertex_normals, faces=self.__triangle_mesh.faces,
                      face_attribute=self.__faces_defect)
        else:
            name = '%d_%.5f' % (N, isoenergy)
            save_data(file=name, vertices=self.__triangle_mesh.vertices,
                      vertices_normal=self.__triangle_mesh.vertex_normals, faces=self.__triangle_mesh.faces,
                      face_attribute=self.__faces_defect)

    def plot_mesh(self):
        M = self.__faces_defect
        K_faces = abs(self.__faces_defect)
        k_min = min(K_faces)
        k_max = max(K_faces)
        K_faces = (K_faces - k_min) / (k_max - k_min)
        K_faces = np.clip(K_faces, self.plot_parameter['clip_min'], self.plot_parameter['clip_max']) / self.plot_parameter['clip_max']

        viridis = cm.get_cmap(None, self.plot_parameter['N_cmap'])
        color = viridis(K_faces)

        plot_surface2(verts=self.__triangle_mesh.vertices, faces=self.__triangle_mesh.faces, color=color)



if __name__ == '__main__':

    A = fermicurve(file='Au60.npy',
                   mesh_parameter={'N': 60, 'N_average': 10, 'isoenergy': 0.0, 'scale': 1.0},
                   plot_parameter={'clip_min': 0, 'clip_max': 1e-2, 'N_cmap': 50}
                   )

    A.create_mash()
    A.plot_mesh()