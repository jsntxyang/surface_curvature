import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import networkx as nx
import math
from Triangle_Mesh import Trim
from random import shuffle
from trimesh.base import Trimesh as TM
from trimesh.smoothing import filter_humphrey
from trimesh.smoothing import filter_laplacian
from skimage.draw import ellipsoid


def read_data(file=None):
    Ry = 13.6
    with open(file, 'r') as f:
        grid = f.readline()
        grid = grid.split()
        grid_x = int(grid[0])
        grid_y = int(grid[1])
        grid_z = int(grid[2])

        type = f.readline()

        nbands = int(f.readline())

        kx = f.readline()
        kx = kx.split()
        for i in range(0, len(kx)):
            kx[i] = float(kx[i])
        kx = np.array(kx)

        ky = f.readline()
        ky = ky.split()
        for i in range(0, len(ky)):
            ky[i] = float(ky[i])
        ky = np.array(ky)

        kz = f.readline()
        kz = kz.split()
        for i in range(0, len(kz)):
            kz[i] = float(kz[i])
        kz = np.array(kz)

        Ek = np.zeros((nbands, grid_x, grid_y, grid_z))
        for i in range(0, nbands):
            for j in range(0, grid_x):
                for l in range(0, grid_y):
                    for m in range(0, grid_z):
                        Ek[i][j][l][m] = float(f.readline())

        K_data = np.zeros((nbands, grid_x, grid_y, grid_z))
        for i in range(0, nbands):
            for j in range(0, grid_x):
                for l in range(0, grid_y):
                    for m in range(0, grid_z):
                        K_data[i][j][l][m] = float(f.readline())

    return kx, ky, kz, Ek * Ry, K_data


def grid_interplot(data):
    shape = data.shape
    Nx = shape[0]
    Ny = shape[1]
    Nz = shape[2]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)

    interpolating_function = RegularGridInterpolator((x, y, z), data, method='linear')

    return interpolating_function


def create_networks(faces):
    G = nx.Graph()
    for i in range(0, len(faces)):
        x0 = faces[i][0]
        x1 = faces[i][1]
        x2 = faces[i][2]
        G.add_edge(x0, x1)
        G.add_edge(x0, x2)
        G.add_edge(x1, x2)
    return G


def create_grid(Nx, Ny, Nz):
    X = np.linspace(0, 1, Nx)
    Y = np.linspace(0, 1, Ny)
    Z = np.linspace(0, 1, Nz)

    coordinate = np.zeros((Nx, Ny, Nz, 3))

    for i in range(0, Nx):
        for j in range(0, Ny):
            for k in range(0, Nz):
                vector = np.array([X[i], Y[j], Z[k]])
                coordinate[i][j][k][0] = vector[0]
                coordinate[i][j][k][1] = vector[1]
                coordinate[i][j][k][2] = vector[2]
    return coordinate


def constrain(verts, faces, fun):
    v_map = np.zeros(len(verts), dtype=int)
    v_del_list = []
    counter = 0
    for i in range(0, len(verts)):
        if not fun(verts[i][0], verts[i][1], verts[i][2]):
            v_map[i] = -1
            v_del_list += [i]
        else:
            v_map[i] = counter
            counter += 1
    new_verts = np.delete(verts, v_del_list, 0)
    faces_del_list = []

    for i in range(0, len(faces)):
        v0 = faces[i][0]
        v1 = faces[i][1]
        v2 = faces[i][2]
        if v_map[v0] == -1:
            faces_del_list += [i]
            continue
        else:
            faces[i][0] = v_map[v0]
        if v_map[v1] == -1:
            faces_del_list += [i]
            continue
        else:
            faces[i][1] = v_map[v1]
        if v_map[v2] == -1:
            faces_del_list += [i]
            continue
        else:
            faces[i][2] = v_map[v2]
    new_faces = np.delete(faces, faces_del_list, 0)

    return new_verts, new_faces


def surface_list(verts, faces):
    S_face = np.zeros(len(faces))
    for i in range(0, len(faces)):
        x0 = faces[i][0]
        x1 = faces[i][1]
        x2 = faces[i][2]
        a1 = verts[x0] - verts[x1]
        a2 = verts[x0] - verts[x2]
        a3 = verts[x1] - verts[x2]
        S = 1 / 2 * np.linalg.norm(np.cross(a1, a2)) * 1e5
        M = np.array([a1, a2, a3])
        x = np.linalg.det(M)
        S_face[i] = abs(S)

    return S_face


def gaussian_curvature(verts, faces, face_of_v, S_face):
    TM = Trim(vertices=verts, faces=faces)

    def gaussian(V0):
        if V0 not in TM.nodes:
            return 0
        if TM.is_boundry_vertex(V0):
            neighbor = list(TM.neighbors(V0))
            K = 0
            counter = 0
            for u in neighbor:
                if TM.is_boundry_vertex(u):
                    continue
                else:
                    K += gaussian(u)
                    counter += 1
            if counter == 0:
                for v in neighbor:
                    neighbor_v = list(TM.neighbors(v))
                    for w in neighbor_v:
                        if TM.is_boundry_vertex(w):
                            continue
                        else:
                            K += gaussian(w)
                            counter += 1
            if counter == 0:
                for v in neighbor:
                    neighbor_v = list(TM.neighbors(v))
                    for w in neighbor_v:
                        neighbor_w = list(TM.neighbors(w))
                        for t in neighbor_w:
                            if TM.is_boundry_vertex(t):
                                continue
                            else:
                                K += gaussian(t)
                                counter += 1
            return K / (counter + 0.1)
        else:
            K = 0
            s = 0
            currentv = verts[V0]
            for u in face_of_v[V0]:
                triangle = faces[u, :]
                a = []
                for k in range(0, 3):
                    if triangle[k] != V0:
                        a += [triangle[k]]
                p1 = verts[a[0]]
                p2 = verts[a[1]]
                #a = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
                #b = np.sqrt((p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2) * (p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2))
                # b = (np.linalg.norm(p2 - currentv) * np.linalg.norm(p1 - currentv))
                cosx = np.dot(p1 - currentv, p2 - currentv) / (
                            np.linalg.norm(p2 - currentv) * np.linalg.norm(p1 - currentv))

                if (cosx > 1) and (abs(cosx - 1) <= 1e-4):
                    cosx = 1
                if (cosx < -1) and (abs(cosx + 1) <= 1e-4):
                    cosx = -1

                K += math.acos(cosx)
                s += S_face[u]
            if s == 0:
                K = 0
            else:
                K = abs((np.pi * 2 - K) / s)

            return K

    K_verts = np.zeros(len(verts))
    for i in range(0, len(verts)):
        K_verts[i] = gaussian(i)
    K_verts = average(K_verts, TM=TM)
    return average(K_verts, TM=TM)
    #return K_verts


'''
    def gaussian(V0):
        if TM.is_boundry_vertex(V0):
            return 0
        else:
            return 1

    for i in range(0, len(verts)):
        K_verts[i] = gaussian(i)

    return K_verts
'''


def average(K_verts, TM):
    # TM = Trim(vertices=verts, faces=faces)
    x = [i for i in range(0, len(K_verts))]
    shuffle(x)

    for i in range(0, len(x)):
        index = x[i]
        if index not in TM.nodes:
            continue
        neighbor = list(TM.neighbors(index))
        A = [K_verts[u] for u in neighbor]
        K_verts[index] = sum(A) / len(A)

    return K_verts


def in_bz(x, y, z):
    l = 2.6 / 3
    if not x + y + z <= 3 / 4 * l:
        return False
    if not x + y + z >= -3 / 4 * l:
        return False
    if not x + y - z <= 3 / 4 * l:
        return False
    if not x + y - z >= -3 / 4 * l:
        return False
    if not x - y + z <= 3 / 4 * l:
        return False
    if not x - y + z >= -3 / 4 * l:
        return False
    if not x - y - z <= 3 / 4 * l:
        return False
    if not x - y - z >= -3 / 4 * l:
        return False
    if not x <= 1 / 2 * l:
        return False
    if not x >= -1 / 2 * l:
        return False
    if not y <= 1 / 2 * l:
        return False
    if not y >= -1 / 2 * l:
        return False
    if not z <= 1 / 2 * l:
        return False
    if not z >= -1 / 2 * l:
        return False

    return True


def plot_surface(verts, faces, color, fig_size=(10, 10)):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor(color)
    # mesh.set_edgecolor('b')
    ax.add_collection3d(mesh)

    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("$k_z$")
    size = 0.5
    ax.set_xlim(-size, size)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(-size, size)  # b = 10
    ax.set_zlim(-size, size)  # c = 16
    '''
    txt = [str(i) for i in range(0, len(verts))]
    X = verts[:, 0]
    Y = verts[:, 1]
    Z = verts[:, 2]
    for i in range(0, len(X)):
        ax.text(X[i], Y[i], Z[i], s=txt[i])
    '''
    # plt.tight_layout()
    plt.show()


def car_coordinate(Ek):
    N = Ek.shape[0]
    C_Ek = np.zeros((N, N, N))
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                #C_Ek[i][j][k] = Ek[(j + k) % N][(k + i) % N][(i + j) % N]
                C_Ek[i][j][k] = Ek[(k - i) % N][(k + j) % N][(-i + j) % N]
    return C_Ek


if __name__ == '__main__':
    Kx, Ky, Kz, Ek, K_data = read_data(file='vfermi2.frmsf')
    origional_data = car_coordinate(Ek[0, :, :, :])
    fun = grid_interplot(origional_data)
    N = 60
    coordinate = create_grid(N, N, N)
    data = fun(coordinate)
    '''
    NewK_x = (Ky + Kz)
    NewK_y = (Kx + Kz)
    NewK_z = (Kx + Ky)
    '''
    NewK_x = -(Kx + Kz)
    NewK_y = (Ky + Kz)
    NewK_z = (Kx + Ky)
    verts, faces, normals, values = measure.marching_cubes(data, 0.00)
    for i in range(0, len(verts)):
        verts[i] = (verts[i][0] * (NewK_x / N) + verts[i][1] * (NewK_y / N) + verts[i][2] * (NewK_z / N)) - (NewK_x + NewK_y + NewK_z) / 2
    verts, faces = constrain(verts, faces, in_bz)
    '''
    mesh = TM(vertices=verts, faces=faces, normals=normals)
    mesh = filter_laplacian(mesh)
    verts = mesh.vertices
    faces = mesh.faces
    '''
    S_face = surface_list(verts, faces)
    K_faces = np.zeros(len(faces))
    face_of_v = []
    for i in range(0, len(verts)):
        face_of_v += [[]]

    for i in range(0, len(faces)):
        x0 = faces[i][0]
        x1 = faces[i][1]
        x2 = faces[i][2]
        face_of_v[x0] += [i]
        face_of_v[x1] += [i]
        face_of_v[x2] += [i]

    K_verts = gaussian_curvature(verts=verts, faces=faces, face_of_v=face_of_v, S_face=S_face)

    for i in range(0, len(faces)):
        v0 = faces[i][0]
        v1 = faces[i][1]
        v2 = faces[i][2]
        curvature = (K_verts[v0] + K_verts[v1] + K_verts[v2]) / 3
        K_faces[i] = curvature

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).

    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    k_min = min(K_faces)
    k_max = max(K_faces)
    K_faces = (K_faces - k_min) / (k_max - k_min) + (1e-10)
    K_faces = np.log(K_faces)
    k_min = min(K_faces)
    k_max = max(K_faces)
    K_faces = (K_faces - k_min) / (k_max - k_min)
    K_faces = (np.clip(K_faces, 0.6, 1.0) - 0.6) / 0.4
    # color = [str(u) for u in K_faces]

    viridis = cm.get_cmap(None, 1000)
    color = viridis(K_faces)

    plot_surface(verts=verts, faces=faces, color=color)
