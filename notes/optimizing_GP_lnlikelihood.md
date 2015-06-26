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
$$k(r^2) = \lambda^{-1} \exp\left(-\frac{r^2}{2 l^2}\right)$$ 

# Parametrization for better visualization... 
It is hard to visualize the likelihood surface in the original scale,
so we perform a log transformation
$$ a = \log_{10}(\lambda^{-1})$$
$$ b = \log_{10}(l^2)$$

To have:
$$k(r^2) = 10^a \exp\left(-\frac{r^2}{2 \times 10^b}\right)$$ 

## How new parametrization relates to $\rho$ 
\begin{align}
b &= \log_{10}(l^2) \\
&= \log_{10}(1 / \beta)\\
&= -\log_{10}\left(-\frac{1}{4}\ln \rho\right)
\end{align}


# Note on transformation of variables 
Jacobian needed to preserve the area of integrated PDF  
$$f_y(\vec{y}) = f_x(\vec{x}) |\det(J)| $$ 

Only when the transformed variable is the one that we integrate 
with respect to.

Let 
\begin{equation}
\vec{x} = \left(\begin{array}{c}
  \lambda^{-1} \\
  l^2
\end{array}
\right)
= \left(\begin{array}{c}
B^{a}\\ 
B^b
\end{array}
\right)
\end{equation}
where $l^2$ is the characteristic length.

\begin{align}
f_y(\vec{y}) = f_y(a, b) &= 
f_x(\lambda^{-1}, l^2)\left| \det \left( 
\begin{array}{cc}
\frac{\partial \lambda^{-1}}{\partial a} & 
\frac{\partial \lambda^{-1}}{\partial b} \\
\frac{\partial l^2}{\partial a} & 
\frac{\partial l^2}{\partial b} 
\end{array})
\right)\right| \\
&= 
f_x(\lambda^{-1}, l^2)\left| \det \left( 
\begin{array}{cc}
\frac{\partial \exp(a \ln B)}{\partial a} & 
\frac{\partial \exp(a \ln B)}{\partial b} \\
\frac{\partial \exp(b \ln B)}{\partial a} & 
\frac{\partial \exp(b \ln B)}{\partial b} 
\end{array})
\right)\right| \\
&= f_x(B^a, B^b)(B^a + B^b)\ln B
\end{align}

When we implement the log likelihood in the MCMC we need
\begin{equation}
\ln L_y(a, b) = \ln L_x(B^a, B^b) + \ln (B^a + B^b) + \ln (\ln B)
\end{equation}
we can ignore the last term as it is a constant w.r.t. change in $a$ and $b$.

# The GP lnlikelihood 
A test of if the GP likelihood is computed correctly in `George` is available
in `test_George_kern.py` and a illustrative GP class is written in `GP.py`.

In particular, the expression of the log likelihood of a GP is:
\begin{equation}
  \ln L = -\frac{1}{2}(y^T K^{-1} y + \log(|\det (K)|) + \log(2 \pi))
\end{equation}

