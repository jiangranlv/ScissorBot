import subprocess
import os

if __name__ == "__main__":
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"])
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    os.chdir("external/mesh_to_sdf")
    subprocess.run(["pip", "install", "-e", "."])
    os.chdir("../yourdfpy")
    subprocess.run(["pip", "install", "-e", "."])
    os.chdir("../batch_urdf")
    subprocess.run(["pip", "install", "-e", "."])
    os.chdir("../..")
    subprocess.run(["conda", "install", "-c", "conda-forge", "scikit-sparse==0.4.8"])
