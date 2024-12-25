import pygmsh
import trimesh

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [0.3, 0.0],
            [0.3, 0.3],
            [0.0, 0.3],
        ],
        mesh_size=0.015,
    )
    mesh = geom.generate_mesh()

v = mesh.points
f = mesh.cells[1].data
m = trimesh.Trimesh(vertices=v, faces=f)

with open("test_square30cm_ascii.ply", "bw") as f_obj:
    f_obj.write(trimesh.exchange.ply.export_ply(m, encoding="ascii"))
