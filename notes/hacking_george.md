# Shear_gp progress:   
Hacking George  
02 / 18 / 2015  
Karen Ng 

# (Cython) file structures of George 

# Cpp definitions of classes and methods : `include/kernels.h`   

* added new children class of `ExpSquaredKernel` 
* overrode `value()` methods 
	* simple example by multiplying `999` to original value 

# what gets compiled into a `_cpp` file: `george/kernels.pxd`   

* `if` loop to check `kernel_spec.kernel_type`
* added new `kernel_spec.kernel_type` for new classes 

# Actual Python class definition: `george/kernels.py`   

* added new Python class definition with same arguments as 
those of the children class in `include/kernels.h` 
* added class name for `george.kernels` to see at the `__all__` definition
for proper import 
* only specifies suitable `kernel_type.kernel_spec`

# Covariance matrix from `ExpSquaredKernel` 
<img src="img/hacking_george/ExpSquaredKernel.png" width=65%> </img>

# Overridden covariance matrix of `KappaKappaExpSquaredKernel`
<img src="img/hacking_george/KappaKappaExpSquaredKernel.png" width=65%> </img>

# next steps 
