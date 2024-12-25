        
def generate_paper_obj(output_file_name, vertices_num_x, vertices_num_z, length_x, length_z):
    with open(output_file_name, "w") as f:
        def ik2vid(i, k):
            return i * vertices_num_z + k + 1

        for i in range(vertices_num_x):
            for k in range(vertices_num_z):
                f.write("v {} {} {}\n".format(i * length_x / (vertices_num_x - 1), 0.0,
                                                k * length_z / (vertices_num_z - 1)))
        for i in range(vertices_num_x - 1):
            for k in range(vertices_num_z - 1):
                if i % 2 == k % 2:
                    f.write("f {} {} {}\n".format(ik2vid(i, k),
                            ik2vid(i + 1, k), ik2vid(i + 1, k + 1)))
                    f.write("f {} {} {}\n".format(ik2vid(i, k),
                            ik2vid(i + 1, k + 1), ik2vid(i, k + 1)))
                else:
                    f.write("f {} {} {}\n".format(ik2vid(i, k),
                            ik2vid(i + 1, k), ik2vid(i, k + 1)))
                    f.write("f {} {} {}\n".format(ik2vid(i + 1, k),
                            ik2vid(i + 1, k + 1), ik2vid(i, k + 1)))

def main():
    dx = float(input("dx=?mm")) / 1000
    length_x = float(input("x=?cm")) / 100
    length_z = float(input("z=?cm")) / 100

    assert dx > 0.0
    vertices_num_x = int(length_x / dx + 0.5) + 1
    vertices_num_z = int(length_z / dx + 0.5) + 1
    output_file_name = "vertical_{}x{}_{}mm.obj".format(int(length_x * 100), int(length_z * 100), int(dx * 1000 + 0.5))
    generate_paper_obj(output_file_name, vertices_num_x, vertices_num_z, length_x, length_z)

if __name__ == "__main__":
    main()