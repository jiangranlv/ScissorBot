import subprocess
import multiprocessing
import time
import sys
import argparse


def worker(file_name, s, e, tex_file):
    print("worker start", file_name, s, e, tex_file)
    subprocess.run([
        "blender",
        "--python",
        "policy/cutgym_render.py",
        "-b",
        "-f",
        file_name,
        "-s",
        str(s),
        "-e",
        str(e),
        "--tex_file",
        str(tex_file)
    ])
    print("worker end", file_name, s, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", type=str, required=True)
    parser.add_argument("-s", "--start-ply", type=int, default= 0 )
    parser.add_argument("-p", "--total-ply", type=int, default= 2000)
    parser.add_argument("-n", "--job-number", type=int, default= 4)
    parser.add_argument("--tex_file", type=str, help="Rendering start index.")
    args = parser.parse_args()

    file_name = args.file_name
    total_ply = args.total_ply
    start_ply = args.start_ply
    n = args.job_number

    process_list = []
    for i in range(n):
        s = total_ply // n * i + start_ply
        if i == n - 1:
            e = total_ply + start_ply
        else:
            e = total_ply // n * (i + 1) + start_ply
        p = multiprocessing.Process(
            target=worker, args=(file_name, s, e, args.tex_file))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == "__main__":
    main()
