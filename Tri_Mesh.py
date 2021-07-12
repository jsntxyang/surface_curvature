from trimesh import Trimesh
class Trim(Trimesh):
    def __init__(self, vertices=None, faces=None, face_normals=None, vertex_normals=None, face_colors=None,
                 vertex_colors=None, face_attributes=None, vertex_attributes=None, metadata=None, process=True,
                 validate=False, use_embree=True, initial_cache=None, visual=None, **kwargs):
        Trimesh.__init__(self, vertices=vertices,
                         faces=faces,
                         face_normals=face_normals,
                         vertex_normals=vertex_normals,
                         face_colors=face_colors,
                         vertex_colors=vertex_colors,
                         face_attributes=face_attributes,
                         vertex_attributes=vertex_attributes,
                         metadata=metadata,
                         process=process,
                         validate=validate,
                         use_embree=use_embree,
                         initial_cache=initial_cache,
                         visual=visual,
                         kwargs=kwargs
                         )
