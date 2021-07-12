import numpy as np
from scipy.interpolate import RegularGridInterpolator
'''
This module includes some I/O functions and data pretreatment functions.
The program can read .frmsf files generated from quantumespresso.

###############
#FUNCTION LIST#
###############

%%%%%%%%%%%%%%%%%%%%%%%%                
#read_data(file=None)#
                    
Parameters:
    file(string)--Input .frmsf file

Returns:
    Kx, Ky, Kz, Ek, Vk
    Kx, Ky, Kz(numpy.ndarray ,3)--Unit cell vector of k-space.
    Ek(numpy.ndarray nbands * Nx * Ny * Nz)--Energy at each k-points
    Vk(numpy.ndarray nbands * Nx * Ny * Nz)--Other values at each k-points
    
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%                
#grid_interpolation(data)#
                    
Parameters:
    file(numpy.npy 3D)--Input data for interpolation.

Returns:
    interpolating_function(callable)--Interpolation_functions with x, y, z variables change from 0 to 1.
    
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%                
#unit_grid(Nx, Ny, Nz)#
                    
Parameters:
    Nx, Ny, Nz--Number of grid points in x, y and z directions

Returns:
    coordinate(numpy.ndarray Nx*Ny*Nz*3)--coordinate array.
    
%%%%%%%%%%%%%%%%%%%%%%%%

'''
def read_data(file=None):
    Ry = 13.6
    with open(file, 'r') as f:
        grid = f.readline()
        grid = grid.split()
        Nx = int(grid[0])
        Ny = int(grid[1])
        Nz = int(grid[2])

        type = f.readline()

        nbands = int(f.readline())

        kx = f.readline()
        kx = list(kx.split())
        for i in range(0, len(kx)):
            str_value = kx[i]
            kx[i] = float(str_value)
        kx = np.array(kx)

        ky = f.readline()
        ky = list(ky.split())
        for i in range(0, len(ky)):
            str_value = ky[i]
            ky[i] = float(str_value)
        ky = np.array(ky)

        kz = f.readline()
        kz = list(kz.split())
        for i in range(0, len(kz)):
            str_value = kz[i]
            kz[i] = float(str_value)
        kz = np.array(kz)

        Ek = np.zeros((nbands, Nx, Ny, Nz))
        for i in range(0, nbands):
            for j in range(0, Nx):
                for l in range(0, Ny):
                    for m in range(0, Nz):
                        Ek[i][j][l][m] = float(f.readline())

        Vk = np.zeros((nbands, Nx, Ny, Nz))
        for i in range(0, nbands):
            for j in range(0, Nx):
                for l in range(0, Ny):
                    for m in range(0, Nz):
                        Vk[i][j][l][m] = float(f.readline())

        Ek = Ek * Ry

    return kx, ky, kz, Ek, Vk

def read_bxsf_data(file=None, nband=1):
    flag = False
    with open(file, 'r') as f:
        for i in range(0, 3):
            f.readline()
        A = f.readline().split(':')[1].strip()
        fermi_energy = float(A)
        for i in range(0, 5):
            f.readline()
        tot_bands = int(f.readline().strip())

        if nband > tot_bands:
            raise ValueError('Too much bands!')

        A = f.readline().split()
        Nx = int(A[0])
        Ny = int(A[1])
        Nz = int(A[2])

        A = f.readline().split()
        center = np.zeros(3)
        center[0] = float(A[0])
        center[1] = float(A[1])
        center[2] = float(A[2])

        A = f.readline().split()
        Kx = np.zeros(3)
        Kx[0] = float(A[0])
        Kx[1] = float(A[1])
        Kx[2] = float(A[2])

        A = f.readline().split()
        Ky = np.zeros(3)
        Ky[0] = float(A[0])
        Ky[1] = float(A[1])
        Ky[2] = float(A[2])


        A = f.readline().split()
        Kz = np.zeros(3)
        Kz[0] = float(A[0])
        Kz[1] = float(A[1])
        Kz[2] = float(A[2])


        Ek = np.zeros((Nx, Ny, Nz))
        counter = 0
        while True:
            A = f.readline()
            if not A:
                break
            elif A.find('BAND') != -1:
                if A.find('END') != -1:
                    break
                A = int(A.split(':')[1].strip())
                if A == nband:
                    flag = True
                else:
                    flag = False
                continue
            elif flag:
                A = A.split()
                for i in range(0, len(A)):
                    index_z = counter % Nz
                    index_y = counter % (Nz * Ny) // Nz
                    index_x = counter // (Nz * Ny)
                    counter += 1
                    Ek[index_x][index_y][index_z] = float(A[i])

    Ek = (Ek[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] - fermi_energy)

    return Ek

def read_npy_data(file=None):
    a = 4.07
    if file:
        Ek = np.load(file)
        kx = np.pi * 2 / a * np.array([-1, 1, 1])
        ky = np.pi * 2 / a * np.array([1, -1, 1])
        kz = np.pi * 2 / a * np.array([1, 1, -1])
        return kx, ky, kz, Ek, None
    else:
        return None


def save_data(file=None, vertices=None, vertices_normal=None, faces=None, face_attribute=None):
    Nv = vertices.shape[0]
    Nf = faces.shape[0]

    if Nv != vertices_normal.shape[0]:
        raise IndexError('vertices_normal has different size with vertices')
    if Nf != len(face_attribute):
        raise IndexError('face_attribute has different size with faces')

    with open(file, 'w') as f:
        f.write(str(Nv) + '\t' + str(Nf) + '\n')
        for i in range(0, Nv):
            px = vertices[i][0]
            py = vertices[i][1]
            pz = vertices[i][2]
            nx = vertices_normal[i][0]
            ny = vertices_normal[i][1]
            nz = vertices_normal[i][2]
            f.write('%.8f\t%.8f\t%8f\t%.8f\t%.8f\t%.8f\n' % (px, py, pz, nx, ny, nz))

        for i in range(0, Nf):
            n0 = faces[i][0]
            n1 = faces[i][1]
            n2 = faces[i][2]
            attribute = face_attribute[i]
            f.write('%d\t%d\t%d\t%.8f\n' % (n0, n1, n2, attribute))

    return None

def load_data(file=None):
    with open(file, 'r') as f:
        line = f.readline()
        N = line.split()
        Nv = int(N[0])
        Nf = int(N[1])
        vertices = np.zeros((Nv, 3))
        vertices_normal = np.zeros((Nv, 3))
        faces = np.zeros((Nf, 3))
        faces_attribute = np.zeros(Nf)
        for i in range(0, Nv):
            line = f.readline()
            value = line.split()
            vertices[i][0] = float(value[0])
            vertices[i][1] = float(value[1])
            vertices[i][2] = float(value[2])
            vertices_normal[i][0] = float(value[3])
            vertices_normal[i][1] = float(value[4])
            vertices_normal[i][2] = float(value[5])

        for i in range(0, Nf):
            line = f.readline()
            value = line.split()
            faces[i][0] = int(value[0])
            faces[i][1] = int(value[1])
            faces[i][2] = int(value[2])
            faces_attribute[i] = float(value[3])

    return vertices, faces, vertices_normal, faces_attribute

def grid_interpolation(data):
    shape = data.shape
    Nx = shape[0]
    Ny = shape[1]
    Nz = shape[2]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)

    interpolating_function = RegularGridInterpolator((x, y, z), data, method='linear')

    return interpolating_function

def unit_grid(Nx, Ny, Nz):
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

def int_transfer_coordinate(Ek):
    N = Ek.shape[0]
    C_Ek = np.zeros((N, N, N))
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                C_Ek[i][j][k] = Ek[(j + k) % N][(k + i) % N][(i + j) % N]
    return C_Ek


if __name__ == '__main__':
    read_npy_data(file='B.npy')
