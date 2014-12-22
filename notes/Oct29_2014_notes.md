# Notes from Oct 29, 2014 meeting
We want to relate observables to the 2D PDF of $\psi_s$:
\begin{equation}
(\alpha, \delta, e_1, e_2, \sigma_e) \rightarrow Pr(\psi_s) 
\end{equation}
Note that $e_1$ and $e_2$ are related to $g_1$, $g_2$ not $\gamma_1$,
$\gamma_2$ directly.

## Steps
0. generate Gaussian $\psi$ 
	* fit $\hat{\psi}$
1. Calculate $\gamma$, $\kappa$, from $\psi$
	* fit $\gamma$, $\kappa$
	* need to regularize $\Sigma$ (covariance
	matrix) to ensure smoothness among other physical conditions  

## distance and metric 
$r = \Delta x^T D^{-1} \Delta x$, 
where we have used a Euclidean metric D (to be consistent with Michael's
notation),  
\begin{align}
D = \beta \left(
\begin{array}{cc}
1 & 0 \\
0 & 1 
\end{array}
\right),
\end{align}
with $\beta = -\frac{1}{4} \ln \rho$, but we can choose another metric
later on.


# details of step 0 
\begin{equation} 
\psi_s \sim GP(0, \Sigma), \\
\end{equation}

where $\lambda$ is called precision and $\rho$ the correlation, 
\begin{align}
\Sigma &= \Sigma(\lambda, \rho) \\  
&= \lambda^{-1}K(\rho) + I,
\end{align} 

and we may want a Mat$\'{e}$rn covariance function,
according to Michael,
\begin{align}
K(\lambda, \rho) &= \exp^{-\beta a(\Delta \alpha^2 + \Delta \delta^2)}\\
&=\rho^{4a(\Delta \alpha^2 + \Delta \delta^2)},
\end{align}
with $\rho \in [0, 1]$, 
Note: RA and DEC need to be in units of radians  

Depending the degrees of freedom (d.o.f.) of the Mat$\'{e}$rn function, if 
$d.o.f \rightarrow \infty$, it's just a squared exponential kernel 
\begin{equation}
K(l ; s) = \exp\left(-\frac{s^2}{2 l^2}\right)
\end{equation}
with $s = \sqrt{\Delta \alpha^2 + \Delta \delta^2}$ and $l$ isthe
characteristic length scale over which large fluctuation in the signal occurs.
In our case, the characteristic length is:e
: 
\begin{equation}
l = \frac{1}{\sqrt{2 a\beta}} = \sqrt{-\frac{2}{a \ln \rho}} 
\end{equation}

# technical challenges 
* come up with conditional update rule - make use of [Schur compliments](http://en.wikipedia.org/wiki/Schur_complement)  

# steps 1  
make use of the fact that $\gamma$ and $\kappa$ are derivatives of the
scalar potential $\psi_s$:
\begin{align}
\gamma_1, \gamma_2 &\sim GP(0, \Sigma^{\gamma}_{xx, yy}(x,y))\\
\kappa &\sim GP(0, \Sigma^{\kappa}_{xx, yy}(x,y))
\end{align}
eqns (8) and (9) are in the draft of the paper also.
