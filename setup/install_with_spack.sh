#!/bin/bash 
spack unload python
# cd to the george main directory
# export PYTHONPATH=$PYTHONPATH:/global/u1/k/karenyng/spack/opt/spack/unknown_arch/gcc-4.3/py-george-0.3.1-33tz4ip3rejxulpzheetariy4mdh5mid

python setup.py build_ext -I/global/u1/k/karenyng/eigen-eigen-bdd17ee3b1b3
develop --prefix=/global/u1/k/karenyng/spack/opt/spack/unknown_arch/gcc-4.3/py-george-0.3.1-33tz4ip3rejxulpzheetariy4mdh5mid
