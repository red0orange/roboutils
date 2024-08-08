import open3d as o3d


def create_o3d_coord(T, size=0.6):
    """
    Create a open3d coordinate frame from a 4x4 transformation matrix.
    :param T: 4x4 transformation matrix.
    :param size: size of the coordinate frame.
    :return: open3d.geometry.TriangleMesh
    """
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    coordinate_frame.transform(T)
    return coordinate_frame