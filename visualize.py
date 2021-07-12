import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go


def plot_surface(verts, faces, color, fig_size=(10, 10)):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("$k_z$")
    size = 0.5
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    plt.show()

def plot_surface2(verts, faces, color):
    X = verts[:, 0]
    Y = verts[:, 1]
    Z = verts[:, 2]

    I = faces[:, 0]
    J = faces[:, 1]
    K = faces[:, 2]

    A = go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color='lightpink', opacity=1.0, facecolor=color)
    fig = go.Figure(data=A)
    fig.show()