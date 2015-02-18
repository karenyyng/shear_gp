# Shear_gp progress:   
Hacking George  
02 / 18 / 2015  
Karen Ng 

# (Cython) file structures of George 
```sh
george
|		setup.py
|		README.rst
|_______include
		|______	kernels.h
|_______george	
		|______ kernels.pxd
		|______ kernels.py
```

# Cpp definitions of classes and methods : `include/kernels.h`   
<ul> 
<li class="fragment"> added new children classes of `ExpSquaredKernel` </li> 
<li class="fragment"> overrode `value()` methods  </li> 
<ul>
	<li class="fragment"> simple example by multiplying `999` to
original value </li>
</ul>
</ul>

# what gets compiled into a `_cpp` file: `george/kernels.pxd`   

<ul>
<li class="fragment"> `if` statements to check `kernel_spec.kernel_type`</li> 
<li class="fragment"> added new `kernel_spec.kernel_type` for new classes </li> 
</ul>

# Actual Python class definition: `george/kernels.py`   
<ul>
<li class="fragment"> added new Python class definition with same arguments as 
those of the children class in `include/kernels.h` 
<li class="fragment"> added class name for `george.kernels` to see at the `__all__` definition
for proper import </li> 
<li class="fragment"> 
only specifies suitable `kernel_type.kernel_spec`
</li> 
</ul>

</section>

# Kernel matrix from `ExpSquaredKernel` 
<img src="img/hacking_george/ExpSquaredKernel.png" width=65%> </img>

# Overridden kernel matrix of `KappaKappaExpSquaredKernel`
<img src="img/hacking_george/KappaKappaExpSquaredKernel.png" width=65%> </img>

# next steps for the Cython implementation 
1. implement the `KernelDerivative` class with general methods 
2. implement similar function as my Python prototype code
3. check results against test cases  

# better matrix visualization from Python prototype code 

# benchmark plot from `ExpSquaredKernel`
<img src="img/hacking_george/ExpSqCov.png" width=40%> </img>

points on a line between [0, 0] and [9, 0] inclusive, spacing = 1

# 
<img src="img/hacking_george/plotsKappaKappaExpSquaredKernel.png" width=35%> </img>

<span class="math">
$$Cov ~\left(\kappa, \kappa \right)=\frac{1}{4} (\Sigma_{,1111} +
\Sigma_{,1122} + $$ 
$$\Sigma_{,2211} + \Sigma_{,2222} )$$
</span>

# Recall 
$$
\Sigma_{,x_i x_j y_h y_k} = (\beta^4 X_h X_j X_k X_i -  
$$
$$
\beta^3 (X_j X_i D_{hk} \delta_{hk} + 5 {\rm ~perm.}) + 
$$
$$
\beta^2 (D_{jh} D_{ik}\delta_{jh}\delta_{ik} + 2 {\rm ~perm.})) \Sigma
$$  
<br>  <br> 
where   

$$ X_h = D(\vec{x} - \vec{y})_h$$

# Zero on the diagonal 
<img src="img/hacking_george/plotsKappaGamma1ExpSquaredKernel.png" width=35%> </img>   
$$
Cov(\kappa, \gamma_1) = \frac{1}{4}(
\Sigma_{,1111} + \Sigma_{,2211} 
$$
$$
- \Sigma_{,1122} - \Sigma_{,2222})
$$

# 
<img src="img/hacking_george/plotsKappaGamma2ExpSquaredKernel.png" width=35%> </img>   

$$
Cov(\kappa, \gamma_2) = 0 = \frac{1}{4}(
\Sigma_{,1112} - \Sigma_{,2212} - 
$$
$$\Sigma_{,1121} + \Sigma_{,2221}
)
$$

# $\Sigma(\gamma_1, \gamma_1)$ 
<img src="img/hacking_george/plotsGamma1Gamma1ExpSquaredKernel.png" width=55%> </img>

# 
<img src="img/hacking_george/plotsGamma1Gamma2ExpSquaredKernel.png" width=35%> </img>   

$$
Cov(\gamma_1, \gamma_2) = 0 = \frac{1}{4}(
\Sigma_{,1112} +
$$
$$
\Sigma_{,1121} - \Sigma_{,2212} - \Sigma_{,2221})
$$

# $\Sigma(\gamma_2, \gamma_2)$ 
<img src="img/hacking_george/plotsGamma2Gamma2ExpSquaredKernel.png" width=55%> </img>


# To do for the prototype code: 
* worked out preliminary diagnostics for all the Cov matrices 
