import subprocess
import multiprocessing
import time
import sys
import argparse


def worker(python_script, file_name, s, e):
    print("worker start", python_script, file_name, s, e)
    subprocess.run([
        "blender",
        "--python",
        python_script,
        "-b",
        "-f",
        file_name,
        "-s",
        str(s),
        "-e",
        str(e),
    ])
    print("worker end", python_script, file_name, s, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--python-script", type=str,
                        required=True, help="Python script to render.")
    parser.add_argument("-f", "--file-name", type=str, required=True,
                        help="Folder where contains your ply models.")
    parser.add_argument("-s", "--start-ply", type=int, required=True)
    parser.add_argument("-p", "--total-ply", type=int, required=True)
    parser.add_argument("-n", "--job-number", type=int, required=True)
    args = parser.parse_args()

    python_script = args.python_script
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
            target=worker, args=(python_script, file_name, s, e))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == "__main__":
    main()
