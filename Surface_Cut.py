import numpy as np

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

def remove_surface(verts, faces, fun, a=4.07, scale=1.0):
    remove = np.zeros(len(verts), dtype=bool)
    v_del_list = []
    for i in range(0, len(verts)):
        if not fun(verts[i][0], verts[i][1], verts[i][2], a, scale):
            v_del_list += [i]
            remove[i] = True
        else:
            pass
    faces_del_list = []

    for i in range(0, len(faces)):
        v0 = faces[i][0]
        v1 = faces[i][1]
        v2 = faces[i][2]

        if remove[v0]:
            faces_del_list += [i]
            continue
        elif remove[v1]:
            faces_del_list += [i]
            continue
        elif remove[v2]:
            faces_del_list += [i]
            continue
        else:
            pass

    return faces_del_list
