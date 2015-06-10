Check out our [Trello project
page](https://trello.com/b/gvclhujL/shear-gp) for more info.

# Table Of Content
<a id='TOC'></a>
* [Unit tests](#Tests)
* [Package prerequisites](#package_prerequisites)
* [Shear_gp file organizations](#shear_gp_file_organization)
* [File organization of George](#George_organization)
* [Setting up George](#setting_up_george)

# Unit tests 
<a id='Tests'></a>


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

# Package prerequisites 
<a id='package_prerequisites'></a>
See `py_package_dependencies.txt` for prerequisite packages to run this code.
Here is a
[post](http://karenyyng.github.io/using-virtualenv-for-safeguarding-research-project-dependencies.html)
on how to set up a `virtualenv` for this project. 

# Shear_gp file organization 

<a id='shear_gp_file_organization'></a>
* modules with functions for analyzing code are in this directory
* code for calling the functions are in the subdirectories

## File organization of `George` 
<a id='George_organization'></a>
[Link to my fork of
George](https://github.com/karenyyng/george/blob/master/document/file_organization.md)


## Note when setting up `George`
<a id='setting_up_george'></a>
Don't compile `HODLR`, clone it and rename folder into `hodlr`
in the github designated location.
Then just do:
```functions 
$ python setup.py develop 
```

# Note about `ipynb` magic
<a id='ipynb_autoreload'></a>
The `autoreload` magic automatically updates function definitions from imported 
modules, this requires the magic function `%autoreload`, 
if you are not Karen Ng you need to run 

    %load_ext autoreload 
    
first, before you can use   
   
    %autoreload

in the notebook.
