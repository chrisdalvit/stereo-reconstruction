import platform
from invoke import task

@task
def build_omp_block_matching(c):
    if platform.system() == "Darwin":
        c.run("gcc -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -shared -O3 -o c/block_matching.so -fPIC c/block_matching.c")
    elif platform.system() == "Linux":
        c.run("gcc -fopenmp -shared -O3 -o c/block_matching.so -fPIC c/block_matching.c")

@task
def build_block_matching(c):
    c.run("gcc -shared -O3 -o c/block_matching.so -fPIC c/block_matching.c")