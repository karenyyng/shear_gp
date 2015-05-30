Check out our [Trello project
page](https://trello.com/b/gvclhujL/shear-gp) for more info.


# tests 
You can use `pytest` to run the tests
```
$ py.test -v test_kern_deriv.py
```
this automatically detects all the files and test functions with `test_` prefixes.

If you need to debug the tests / code, instead run:
```
$ py.test --pdb -v test_kern_deriv.py
```
to trigger the python debugger `pdb` post mortem.



# prerequisites 
See `py_package_dependencies.txt` for prerequisite packages to run this code.
Here is a
[post](http://karenyyng.github.io/using-virtualenv-for-safeguarding-research-project-dependencies.html)
on how to set up a `virtualenv` for this project. 

# file structures 
* functions for analyzing code are in this directory
* code for calling the functions are in the subdirectories

## Note when setting up `George`
Don't compile `HODLR`, clone it and rename folder into `hodlr`
in the github designated location.
Then just do:
```
$ python setup.py develop 
```

# note about ipynb magic
The autoreload magic automatically updates function definitions from imported modules, 
this requires the magic function %autoreload, if you are not Karen Ng you need to run 

    %load_ext autoreload 
    
first, before you can use   
   
    %autoreload

in the notebook
