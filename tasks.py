from invoke import task

@task
def build_omp_block_matching(c):
    c.run("gcc -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -shared -O3 -o c/block_matching.so -fPIC c/block_matching.c")
    
@task
def build_block_matching(c):
    c.run("gcc -shared -O3 -o c/block_matching.so -fPIC c/block_matching.c")
    
@task
def build_sgm(c):
    c.run("gcc -shared -O3 -o c/sgm.so -fPIC c/sgm.c")