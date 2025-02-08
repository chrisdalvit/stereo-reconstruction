# Point Cloud Reconstruction from Stereo Images

After cloning the git repository run the following commands in the project root folder
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Now you can run the implemented stereo algorithms with 
```
python main.py --method sgm --data images/cones/
```
you can append the ```--subpixel``` flag to use subpixel interpolation. The ```--method``` argument accepts the following values ```[bm, sgm, opencv]```.
At the beginning of the ```main.py``` you can find a full list of arguments. A PNG of the disparity map and the .ply file of the point cloud are stored in the ```output``` folder.

Before testing the implementation of blockmatching in C you first have to create the shared objects from the C files in the ```c``` folder. You can do this by running
```bash
invoke build-block-matching
```
or for building the C algorithm with OpenMP run
```bash
invoke build-omp-block-matching
```
Both compilation steps have been tested on MacOS 13.4.1 and Ubuntu 22.04. If you want to run the project on windows or change the path to OpenMP you have to modify the command in ```task.py```.

After creating the shared objects (.so files) you can run 
```
python main.py --method bm --language c
```

To **visualize** the generated point cloud run 
```
python render.py --file output/<YOUR_PLY_FILE>
```