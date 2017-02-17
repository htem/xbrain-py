# xbrain-py
__python tools for segmenting X-ray brain volumes__

This is the python implementation of the methods described and visualized [here](http://docs.neurodata.io/xbrain). If you use any of these tools, please cite our [paper](http://arxiv.org/pdf/1604.03629v2.pdf). If you have any questions, please contact Eva Dyer at Eva.L.Dyer {at} gmail {dot} com.

### Getting started...
___(Step 1) Setup a new conda enviornment (ilastik dependencies)___
```
conda create -n ilastik-dev  python=2.7
source activate ilastik-dev 

conda install -c ilastik ilastik-everything-no-solvers
conda create -n ilastik-devel -c ilastik ilastik-everything-no-solvers
conda config --add channels conda-forge
conda install mahotas
pip install ndparse
#conda remove tifffile
#conda install -c ilastik  tifffile=0.4.1
conda install ipython
conda install jupyter
pip install tifffiles

source deactivate
```
Now each time you run ilastik, you need to enter into this environment. To start the environment and launch a notebook, use the following (terminal) command:
```
source activate ilastik-dev
jupyter notebook
```

___(Step 2) Run one of our example notebooks!___
* Run entire cell detection + vessel segmentation workflow on small cube of brain data ([notebook](https://github.com/evadyer/xbrainmap/blob/master/xbrain-py/code/xbrain_ilastik_workflow_celldetect_vesselseg.ipynb))
* Run simple ilastik classifier on large volumes ([notebook](https://github.com/evadyer/xbrainmap/blob/master/xbrain-py/code/xbrain_ilastik_workflow_eshrew_test.ipynb)) 

### Contributors
* Eva Dyer ([@evadyer](http://github.com/evadyer))
* Will Gray Roncal ([@willgray](http://github.com/willgray))
* Mehdi Tondravi ([@anlmehdi](http://github.com/anlmehdi))

