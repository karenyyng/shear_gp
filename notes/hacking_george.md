# Shear_gp progress:   
Hacking George  
02 / 18 / 2015  
Karen Ng 

# (Cython) file structures of George 

# Cpp definitions of classes and methods : `include/kernels.h`   

* added new children class of `ExpSquaredKernel` 
* overrode `value()` methods 
	* simple example by multiplying `999` to original value 

# Interface between Python and Cpp header: `george/kernels.pxd`   

* `if` loop to check `kernel_spec.kernel_type`
* added new `kernel_spec.kernel_type = 10` for new class 

# Actual Python class definition: `george/kernels.py`   

* added new Python class definition with same arguments as 
those of the children class in `include/kernels.h` 
* added class name for `george.kernels` to see at the `__all__` definition
for proper import 
* only specifies suitable `kernel_type.kernel_spec`

# Covariance matrix from `ExpSquaredKernel` 
<img src="img/hacking_george/ExpSquaredKernel.png" width=65%> </img>

# Covariance matrix from overridden method of `KappaKappaExpSquaredKernel`
<img src="img/hacking_george/KappaKappaExpSquaredKernel.png" width=65%> </img>


