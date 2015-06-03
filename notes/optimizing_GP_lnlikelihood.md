# Optimizing the ln likelihood function in George
author: Karen Y. Ng

# Kernel used: 
$$
k(r) = \Sigma(r^2) + c\delta_{ij}
$$
where $\Sigma(r^2)$ is `ExpSquaredKernel` or its derivatives.  
The second term is the `WhiteKernel` in George  

# Parametrization used in `George`
See `george_examples/basic_properties_of_george.ipynb`
$$k(r^2) = \lambda^{-1} \exp(-\frac{r^2}{2 l^2})$$ 

# Transformation of variables 
Jacobian needed to preserve the area of integrated PDF  
$$f_y(\vec{y}) = f_x(\vec{x}) |\det(J)| $$ 

Only when the transformed variable is the one that we integrate 
with respect to.

Since our likelihood can be written as 
$$L(\lambda^{-1}, \beta| \vec{x}) = P(\vec{x} | \lambda^{-1}, \beta)$$ 
it shows that we are not actually integrating w.r.t. our parameters but our
variables $\vec{x}$. No Jacabian is needed.

# Optimizing the GP lnlikelihood 

