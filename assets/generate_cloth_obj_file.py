n = int(input("n=?"))
method = int(input(
    r"""method=?
1:
-----
|\|\|
-----
|\|\|
-----
2:
-----
|/|\|
-----
|\|/|
-----
"""
))
vertices_num_x = n
vertices_num_y = n
length_x = 0.3
length_y = 0.3
output_file_name = "simple_cloth_{}.obj".format(n)

with open(output_file_name, "w") as f:
    def ij2vid(i, j):
        return i * vertices_num_y + j + 1
    if method == 1:
        for i in range(vertices_num_x):
            for j in range(vertices_num_y):
                f.write("v {} {} {}\n".format(i * length_x / (vertices_num_x - 1),
                                              j * length_y / (vertices_num_y - 1), 0))
        for i in range(vertices_num_x - 1):
            for j in range(vertices_num_y - 1):
                f.write("f {} {} {}\n".format(ij2vid(i, j),
                        ij2vid(i + 1, j), ij2vid(i + 1, j + 1)))
                f.write("f {} {} {}\n".format(ij2vid(i, j),
                        ij2vid(i + 1, j + 1), ij2vid(i, j + 1)))

    elif method == 2:
        for i in range(vertices_num_x):
            for j in range(vertices_num_y):
                f.write("v {} {} {}\n".format(i * length_x / (vertices_num_x - 1),
                                              j * length_y / (vertices_num_y - 1), 0.0))
        for i in range(vertices_num_x - 1):
            for j in range(vertices_num_y - 1):
                if i % 2 == j % 2:
                    f.write("f {} {} {}\n".format(ij2vid(i, j),
                            ij2vid(i + 1, j), ij2vid(i + 1, j + 1)))
                    f.write("f {} {} {}\n".format(ij2vid(i, j),
                            ij2vid(i + 1, j + 1), ij2vid(i, j + 1)))
                else:
                    f.write("f {} {} {}\n".format(ij2vid(i, j),
                            ij2vid(i + 1, j), ij2vid(i, j + 1)))
                    f.write("f {} {} {}\n".format(ij2vid(i + 1, j),
                            ij2vid(i + 1, j + 1), ij2vid(i, j + 1)))
