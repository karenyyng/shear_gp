---
title: "Notes - A GP model for shear fields"
author: "Karen Y. Ng"
date: "December 19, 2014"
output: ioslides_presentation
---

# Our Gaussian process "model" for the  projected lensing potential 
\begin{align}
\psi \sim N(0, \Sigma(\vec{x}, \vec{y}))
\end{align}
which is a scalar field evaluated at the positions $x_i = (\vec{x_1},
\vec{x_2})_i$ or $y_i = (\vec{y_1}, \vec{y_2})_i$  where we have data points. 
As usual, first column,
$x_1$ or $y_1$ is the first spatial
dimension, $x_2$ or $y_2$ is the second one, the i-th row correspond to spatial
coordinates of the i-th data point.  
$\vec{x}$ and $\vec{y}$ are the same but we call them different names for
denoting their location in the covariance matrix ....

The convergence and shear,  $\kappa, \gamma_1, \gamma_2$ are the 2nd spatial
derivatives of the lensing potential.
The subscripts in these WL equations correspond to the spatial
coordinates $x_1, x_2$ NOT the observation numbers 
i.e. i, j = 1, 2, ..., n observations
\begin{align*}
\kappa &= \frac{1}{2}tr(\psi_{,ij})\\ 
&= \frac{1}{2} (\psi_{,11} + \psi_{,22})\\ 
&=
\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x_1^2} +
\frac{\partial^2 \psi}{\partial x_2^2 }\right)
\end{align*}

\begin{align*}
\gamma_1 &= \frac{1}{2} (\psi_{,11} - \psi_{,22})\\ 
&=\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x_1^2} - 
\frac{\partial^2 \psi}{\partial x_2^2}\right)
\end{align*}

\begin{align*}
\gamma_2 &= \frac{1}{2} (\psi_{,12} + \psi_{,21})\\ 
&=\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x_1 \partial
x_2} + 
\frac{\partial^2 \psi}{\partial x_2 \partial x_1}\right)
\end{align*}




# Covariances of the required functions 
Note that $\psi, \kappa$ and $\gamma$ are scalar fields. 
However, we are evaluating them at the locations of the data points
$(x_1, x_2)_i$, therefore, when we are writing down the shorthand for the
m, n subscripts below, we mean, we first take the spatial derivatives of
those scalar field(s) with
respect to $x_1$ or $x_2$, then evaluate them at the $m-th$ or $n-th$
position $(x_1,
x_2)_m$.
The spatial derivatives are represented as follows: 
\begin{equation*}
\psi_{,1} = \frac{\partial \psi}{\partial x_1} 
\end{equation*}
etc. with a comma in the subscript. 

Also note expectation and derivative are both linear operators, so we can
exchange their positions (and try not to let mathematicians read this and
shoot us)
\begin{align*}
&{\rm Cov}_{m,n}(\kappa(\vec{x}), \kappa(\vec{y}))\\ 
&= \mathbb{E}\left[ 
(\kappa - \mathbb{E}[\kappa])|_m 
(\kappa - \mathbb{E}[\kappa])|_n 
\right]\\
&= \mathbb{E}\left[ 
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1^2} + \frac{\partial^2}{\partial x_2^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1^2} + \frac{\partial^2}{\partial x_2^2}
\right) \psi 
\right]
\right]\bigg|_m
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1^2} + \frac{\partial^2}{\partial y_2^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1^2} + \frac{\partial^2}{\partial y_2^2}
\right) \psi 
\right]
\right]\bigg|_n
\right]\\
&=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x_1^2} +
\frac{\partial^2}{\partial x_2^2}
\right)\bigg|_m
[\psi - \mathbb{E}[\psi]]|_m
\left(
\frac{\partial^2}{\partial y_1^2} +
\frac{\partial^2}{\partial y_2^2}
\right)\bigg|_n
[\psi - \mathbb{E}[\psi]]|_n
\right]\\
&= \frac{1}{4}
\left(
\left(\frac{\partial^2}{\partial x_1^2}\right)\bigg|_m 
\left(\frac{\partial^2}{\partial y_1^2}\right)\bigg|_n + 
\left(\frac{\partial^2}{\partial x_1^2}\right)\bigg|_m 
\left(\frac{\partial^2}{\partial y_2^2}\right)\bigg|_n +
\left(\frac{\partial^2}{\partial x_2^2}\right)\bigg|_m 
\left(\frac{\partial^2}{\partial y_1^2}\right)\bigg|_n + 
\left(\frac{\partial^2}{\partial x_2^2}\right)\bigg|_m 
\left(\frac{\partial^2}{\partial y_2^2}\right)\bigg|_n  
\right) \Sigma_{mn} \\
&= \frac{1}{4}\left(
\frac{\partial^2}{\partial x_1^2} \frac{\partial^2}{\partial y_1^2} + 
\frac{\partial^2}{\partial x_1^2} \frac{\partial^2}{\partial y_2^2} +  
\frac{\partial^2}{\partial x_2^2} \frac{\partial^2}{\partial y_1^2} + 
\frac{\partial^2}{\partial x_2^2} \frac{\partial^2}{\partial y_2^2}  
\right) \Sigma_{mn} 
\end{align*}

Now I have dropped the (m,n) subscripts and the following subscripts
correspond to the spatial dimensions, the first two subscripts correspond
to spatial derivatives w.r.t. x and evaluated for the m-th data points, the last two
correspond to spatial derivatives w.r.t. y and evaluated for the n-th data
points. 
\begin{align}
{\rm Cov}(\kappa(\vec{x}), \kappa(\vec{y}))
&= \frac{1}{4}\left(
\Sigma_{,1111} + \Sigma_{,1122} + \Sigma_{,2211} + \Sigma_{,2222}
\right)
\end{align}

Similarly,
\begin{align*}
&{\rm Cov}_{mn}(\gamma_1(\vec{x}), \gamma_1(\vec{y})) \\
&= \mathbb{E}\left[ 
(\gamma_1- \mathbb{E}[\gamma_1])|_m
(\gamma_1- \mathbb{E}[\gamma_1])|_n 
\right]\\
&= \mathbb{E}\left[ 
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1^2} - \frac{\partial^2}{\partial x_2^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1^2} - \frac{\partial^2}{\partial x_2^2}
\right) \psi 
\right]
\right]\bigg|_m
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1^2} - \frac{\partial^2}{\partial y_2^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1^2} - \frac{\partial^2}{\partial y_2^2}
\right) \psi 
\right]
\right]\bigg|_n
\right]\\
&=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x_1^2} - 
\frac{\partial^2}{\partial x_2^2}
\right)\bigg|_i
[\psi - \mathbb{E}[\psi]]|_m
\left(
\frac{\partial^2}{\partial y_1^2} - 
\frac{\partial^2}{\partial y_2^2}
\right)\bigg|_j
[\psi - \mathbb{E}[\psi]]|_n
\right]\\
&= \frac{1}{4}\left(
\frac{\partial^2}{\partial x_1^2} \frac{\partial^2}{\partial y_1^2} - 
\frac{\partial^2}{\partial x_1^2} \frac{\partial^2}{\partial y_2^2} -  
\frac{\partial^2}{\partial x_2^2} \frac{\partial^2}{\partial y_1^2} + 
\frac{\partial^2}{\partial x_2^2} \frac{\partial^2}{\partial y_2^2}  
\right) \Sigma_{mn} \\
\end{align*}

\begin{align}
{\rm Cov}(\gamma_1(\vec{x}), \gamma_1(\vec{y}))&= \frac{1}{4}\left(
\Sigma_{,1111} - \Sigma_{,1122} - \Sigma_{,2211} + \Sigma_{,2222}
\right)
\end{align}

And, 


\begin{multline*}
{\rm Cov}_{mn}(\gamma_2(\vec{x}), \gamma_2(\vec{y})) 
= \mathbb{E}\left[ 
(\gamma_2- \mathbb{E}[\gamma_2])|_m 
(\gamma_2- \mathbb{E}[\gamma_2])|_n 
\right]\\
= \mathbb{E}
\left[
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1 \partial x_2} + 
\frac{\partial^2}{\partial x_2 \partial x_1}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x_1 \partial x_2} + 
\frac{\partial^2}{\partial x_2 \partial x_1}
\right) \psi 
\right]
\right]\bigg|_m \right.\\
\left.
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1 \partial y_2} + 
\frac{\partial^2}{\partial y_2 \partial y_1}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial y_1 \partial y_2} + 
\frac{\partial^2}{\partial y_2 \partial y_1} 
\right) \psi 
\right]
\right]\bigg|_n
\right]
\end{multline*}

\begin{align*}
{\rm Cov}_{mn}(\gamma_2(\vec{x}), \gamma_2(\vec{y})) 
=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x_1 \partial x_2} + 
\frac{\partial^2}{\partial x_2 \partial x_1}
\right)\bigg|_m
[\psi - \mathbb{E}[\psi]]|_m
\left(
\frac{\partial^2}{\partial y_1 \partial y_2} + 
\frac{\partial^2}{\partial y_2 \partial y_1}
\right)\bigg|_n
[\psi - \mathbb{E}[\psi]]|_n
\right]
\end{align*}

\begin{align}
{\rm Cov}(\gamma_2(\vec{x}), \gamma_2(\vec{y}))&= \frac{1}{4}\left(
\Sigma_{,1212} + \Sigma_{,1221} + \Sigma_{,2112} + \Sigma_{,2121}
\right)
\end{align}

\begin{align}
{\rm Cov}(\kappa(\vec{x}), \gamma_1(\vec{y})) &= \frac{1}{4}\left(
\Sigma_{,1111} + \Sigma_{,2211} - \Sigma_{,1122} - \Sigma_{,2222}
\right)
\end{align}

\begin{align}
{\rm Cov}(\kappa(\vec{x}), \gamma_2(\vec{y})) &= \frac{1}{4}\left(
\Sigma_{,1112} + \Sigma_{,2212} + \Sigma_{,1121} + \Sigma_{,2221}
\right)
\end{align}

\begin{align}
{\rm Cov}(\gamma_1(\vec{x}), \gamma_2(\vec{y})) &= \frac{1}{4}\left(
\Sigma_{,1112} + \Sigma_{,1121} - \Sigma_{,2212} - \Sigma_{,2221}
\right)
\end{align}

# The squared exponential covariance function 
\begin{align}
\Sigma(r^2; \lambda, \rho) = \lambda^{-1} \exp\left( -
\frac{\beta}{2} r^2 \right)
\end{align}
where $\beta = - 1/4 \ln \rho$, and $0 < \rho < 1$, note $\Sigma$ is an N
$\times$ N matrix and the covariance functions of the derivatives should
have the same dimension. 

# The metric D  
\begin{equation}
r^2 = (\vec{x} - \vec{y})^T D (\vec{x} - \vec{y}) 
\end{equation}

Since we are working in projected (2D) space, D is a 2 $\times$ 2 matrix.
More explicitly, I will use i,j,h,k as subscripts for the spatial
dimensions and m, n for the observation number in the GP model:
\begin{align*}
r^2_{mn} & = (x_{m1} - y_{n1}, x_{m2} - y_{n2})
\left(
\begin{array}{cc}
D_{11} & D_{12} \\ D_{21} & D_{22} 
\end{array}
\right)
\left(
\begin{array}{c}
x_{m1} - y_{n1} \\ x_{m2} - y_{n2} 
\end{array}
\right)
\\
\Sigma_{mn} &= \lambda^{-1} \exp\left( -\frac{\beta}{2} r^2_{mn} \right)
\end{align*}

An example of $r^2$ with an Euclidean metric for a pair of data points,
$\vec{x}_i$ and $\vec{y}_j$ 
would be:
\begin{equation}
r_{mn}^2 = D_{11} (x_{m1}-y_{n1})^2 + D_{22} (x_{m2}-y_{n2})^2 
\end{equation}
assuming diagonal metric.

In the following derivations, it is NOT important to keep the $m, n$
subscripts. We are taking the derivatives w.r.t to the spatial
dimensions, so I will drop the $m, n$ subscripts. 
But keep in mind each
forth derivative of $\Sigma$ with the indices written out should be a
scalar, each is an element in the big $m \times n$ $\Sigma_{,x_i x_j y_h
y_k}$ matrix. 

# Summary: basic derivatives of components, assuming diagonal D  
Derivation of the derivatives for a non-symmetric / non-diagonal D would
give more terms ...

\begin{align*}
\frac{\partial r^2}{\partial x_i} 
&=  \frac{\partial (x_q - y_q)D_{qr} (x_r
- y_r)}{\partial x_i} \\ 
&= \delta_{iq} D_{qr} (x_r - y_r) + (x_q - y_q)\delta_{iq} D_{qr}\\ 
&= 2 D_{ii} (x_i - y_i) \\  
&= 2 [D (\vec{x} - \vec{y})]_i  
\end{align*}

The third line in the above derivation is only true for diagonal D or else
there are other summation terms.

\begin{equation}
r_{,x_i}= 2 [D (\vec{x} - \vec{y})]_i \equiv 2X_i
\end{equation}

Similarly, 
\begin{align*}
\frac{\partial r^2}{\partial y_h} &=  \frac{\partial (x_q - y_q)D_{qr} (x_r
- y_r)}{\partial y_h} \\ 
&= -\delta_{iq} D_{qr} (x_r - y_r) - (x_q - y_q)\delta_{iq} D_{qr}\\ 
&= -2 [D (\vec{x} - \vec{y})]_h
\end{align*}

\begin{equation}
r_{,y_h} = -2 [D (\vec{x} - \vec{y})]_h \equiv -2X_h
\end{equation}

where i = 1, 2 

### Second derivatives of $r^2$ 
\begin{align}
r^2_{,x_i x_j} &= 2 \delta_{ij} D_{ij}\\ 
r^2_{,y_h y_k} &= 2 \delta_{hk} D_{hk}\\ 
r^2_{,x_i y_h} &= -2 \delta_{ih} D_{ih} 
\end{align}

\begin{align}
X_i &= [D(\vec{x} - \vec{y})]_i\\
X_i, x_j &= D \delta_{ij} \\ 
X_i, y_h &= -D \delta_{ih}  
\end{align}

### Derivatives of the kernel
\begin{equation*}
\Sigma = \lambda^{-1} k 
\end{equation*}

\begin{align}
k &= \exp{\left(\frac{-\beta}{2} r^2 \right)}\\
k_{,x_i} &= \frac{-\beta}{2} k r^2_{,x_i} = -\beta k X_i\\ 
k_{,y_h} &= \beta k X_h 
\end{align}

\begin{align}
k_{,x_i x_j} &= \frac{-\beta}{2}(k_{,x_j} r^2_{,x_i} + k r^2_{,x_i x_j})\\
k_{,x_i x_j y_h} &= \frac{-\beta}{2}(k_{,x_j y_h} r^2_{,x_i} + k_{,x_j}
r^2_{,x_i y_h} + k_{,y_h} r^2_{,x_i x_j})\\
k_{,x_i x_j y_h y_k} &= \frac{-\beta}{2} 
(k_{,x_j y_h y_k} r^2_{,x_i} +
k_{,x_j y_h} r^2_{,x_i y_k} + 
k_{,x_j y_k} r^2_{,x_i y_h} + 
k_{,y_h y_k} r^2_{,x_i x_j} )
\end{align}

## Just work on terms that are parts of the 4th kernel derivative 
\begin{align*}
k_{,x_i x_j} &= \frac{\partial}{\partial x_j} (-\beta k X_i)\\
&= -\beta(k_{,x_j X_i} + k X_{i, x_j})\\ 
&= -\beta(-\beta k X_j X_i + k \delta_{ij} D_{ij}) \\ 
&= (\beta^2 X_j X_i - \beta \delta_{ij} D_{ij})k 
\end{align*}

\begin{align*}
k_{,x_i y_h} &= \frac{\partial}{\partial y_h} (-\beta k X_i)\\
&= -\beta(k_{,y_h X_i} + k X_{i, y_h})\\ 
&= -\beta(\beta k X_h X_i - k \delta_{ih} D_{ih}) \\ 
&= -(\beta^2 X_h X_i - \beta \delta_{ih} D_{ih})k 
\end{align*}

\begin{align*}
k_{,y_h y_k} &= \frac{\partial}{\partial y_h} (\beta k X_h)\\
&= \beta(k_{,y_k X_h} + k X_{h, y_k})\\ 
&= \beta(\beta k X_k X_h - k \delta_{hk} D_{hk}) \\ 
&= (\beta^2 X_h X_k - \beta \delta_{hk} D_{hk})k 
\end{align*}

\begin{align}
k_{,x_i x_j} &= (\beta^2 X_j X_i - \beta \delta_{ij} D_{ij})k\\ 
k_{,x_i y_h} &= -(\beta^2 X_h X_i - \beta \delta_{ih} D_{ih})k\\ 
k_{,y_h y_k} &= (\beta^2 X_h X_k - \beta \delta_{hk} D_{hk})k 
\end{align}

## Term 1 of the 4th derivative in eqn (24) 
\begin{align*}
k_{,x_j y_h y_k}
&= \frac{\partial}{\partial y_k} k_{,x_j y_h}\\ 
&= -\frac{\partial}{\partial y_k} (\beta^2 X_h X_j - \beta \delta_{jh} D_{jh})k\\
&= (\beta^2 D_{hk} \delta_{hk} X_j + \beta^2 X_h D_{jk}\delta_{jk})k -
(\beta^2 X_h X_j - \beta D_{jh} \delta_{jh})\beta X_k k
\end{align*}

\begin{align*}
&k_{,x_j y_h y_k} r_{,x_i}^2\\
&= 2[\beta^2 X_j D_{hk} \delta_{hk} + \beta^2 X_h D_{jk}\delta_{jk} +
\beta^2 X_k D_{jh} \delta_{jh} - \beta^3 X_h X_j X_k] X_i k\\ 
&=\boxed{2\beta^2[ 
X_j X_i D_{hk} \delta_{hk} + 
X_h X_i D_{jk} \delta_{jk} + 
X_k X_i D_{jh} \delta_{jh}] k  
- 2\beta^3 X_h X_j X_k X_i k} 
\end{align*}

## Term 2 of the 4th derivative 
\begin{align*}
k_{,x_j y_h} r^2_{,x_i y_k}
&= - (\beta^2 X_h X_j - \beta D_{jh} \delta_{jh}) (-2D_{ik} \delta_{ik}k) \\  
&= \boxed{( 2 \beta^2  X_h X_j D_{ik} \delta_{ik} 
-2 \beta D_{jh} D_{ik} \delta_{jh} \delta_{ik}) k} 
\end{align*}

## Term 3 of the 4th derivative 
This is completely analogous to term 2 except the subscripts are slightly
different
\begin{align*}
k_{,x_j y_k} r^2_{,x_i y_h}
&=\boxed { ( 2 \beta^2  X_k X_j D_{ih} \delta_{ih} 
-2 \beta D_{jk} D_{ih} \delta_{jk} \delta_{ih}) k} 
\end{align*}

## Term 4 of the 4th derivative  
\begin{align*}
k_{,y_h y_k} r^2_{,x_i x_j}
&= (\beta^2 X_k X_h - \beta D_{hk}\delta_{hk})k 2 D_{ij} \delta_{ij}\\ 
&= \boxed{(2\beta^2 X_k X_h D_{ij} \delta_{ij} - 2\beta D_{ij} D_{hk} \delta_{hk}
\delta_{ij})k } 
\end{align*}

## Collect terms of $\Sigma_{,x_i x_j y_h y_k}$ by plugging them in eqn 20 
All the relevant terms are boxed above, 
\begin{align}
\nu_{,x_i x_j y_h y_k} &= (\beta^4 X_h X_j X_k X_i -  
\beta^3 (X_j X_i D_{hk} \delta_{hk} + 5 {\rm perm.}) + \beta^2
(D_{jh} D_{ik}\delta_{jh}\delta_{ik} + 2 {\rm perm.})) \nu \\
&= \gamma \nu
\end{align}

Where $\nu$ is an entry in the matrix $\Sigma$
\begin{equation}
\Sigma  = 
\left(
\begin{array}{ccc}
\nu_{11} & \cdots & \nu_{1n} \\
\vdots & \ddots & \vdots \\
\nu_{n1} & \cdots & \nu_{nn} \\
\end{array}
\right)
\end{equation}

Note that when we evaluate the terms in the parenthesis, they come out to be a
$n \times n$ matrix, and we should multiply those terms to $\Sigma$ using a
[Schur product](http://en.wikipedia.org/wiki/Hadamard_product_(matrices)).

Each spatial derivative result in an extra factor of inverse length in
terms of the units. 
Therefore, the covariance function of the 4th spatial derivative has units
of (inverse length)$^4$.

## Actual Kernel used
It is customary for people to add a white-noise term to the kernel in the form
of:
\begin{equation}
K = \Sigma + \sigma_{noise}^2 I
\end{equation}

## Gradient function for optimizing hyperparameters 
With $\Gamma$ being the matrix containing all the derivative coefficients in
eqn (29), the gradient function can be thought of as
\begin{align}
g(r^2) &= \frac{\partial }{\partial r^2}\Sigma_{,hijk}\\
&= \Gamma \frac{\partial \Sigma}{\partial r^2} + 
\frac{\partial \Gamma}{\partial r^2}\Sigma \\ 
&= -\frac{\beta}{2} \Gamma \Sigma  
\end{align} 
This is due to equation (11) showing how 
\begin{equation*}
\frac{\partial X_i}{\partial r^2} = 0. 
\end{equation*}


## Conditional distribution to learn from $\gamma_1$ or $\kappa$
Our entire covariance matrix with second derivatives give:
\begin{equation}
\Sigma_{,hijk} = \left(\begin{array}{ccc}
\kappa\kappa & \kappa \gamma_1 &  \kappa \gamma_2 \\
\gamma_1 \kappa & \gamma_1  \gamma_1 & \gamma_1 \gamma_2  \\
\gamma_2 \kappa & \gamma_2 \gamma_1 & \gamma_2 \gamma_2  
\end{array}
\right)
\end{equation}

with a data vector: 
\begin{equation}
\vec{d} = \left(
\begin{array}{c}
\vec{x}_{\kappa } \\
\vec{x}_{\gamma_1} \\
\vec{x}_{\gamma_2} \\
\end{array}
\right)
\end{equation}

\begin{equation}
N(\mu_s, \Sigma_s) = N(\mu_{\kappa\kappa}, \Sigma_{\kappa\kappa}|
\vec{d}_{\gamma_1}, \vec{d}_{\kappa} )
\end{equation}


\begin{equation}
\Sigma_s = \Sigma_{\kappa \kappa} - \Sigma_{\kappa \gamma_1} 
\Sigma_{\gamma_1\gamma_1}^{-1} 
\Sigma_{\kappa \gamma_1} 
\end{equation}

## Implementation details 

### Hard coded member variables that should have at most ONE member copy 
* `__ix_list__` = actual subscripts on the R.H.S. of eqn. (2 - 7), 4 $\times$ 4
    in dimension
* `__term_signs__` = signs of the terms on the R.H.S. of (2 - 7), 4 $\times$ 1
    in dimension    
* `__comb_B_ix__` = actual permutation of each of the 4 rows (variations) of `__ix_list__`  after taking the order
    represented by `__pair__of_B_indices__` into account, 6 $\times$ 4 in dimension (we have 6 terms of type B, each term has 4 subscripts).
In conclusion, `__comb_B_ix__` is going to be 6 $\times$ 4 by 4, i.e. 24 by 4. 

* `__comb_C_ix__` = actual permutation of each of the 4 rows (variation) of `__ix_list__`  after taking the order represented by `__pair__of_C_indices__` into account, 3 $\times$ 4 in
    dimension (we have 3 terms of type C, each term has 4 subscripts) 
In conclusion, `__comb_C_ix__` is going to be 3 $\times$ 4 by 4, i.e. 12 by 4. 

#### Miscellaneous 
* the "distance matrix" $X_1$ and $X_2$ should be precomputed / distributed a priori
and called when needed.


#### Within the virtual class `DerivativeExpSquaredKernel`   
The following should only have one copy (per instance)   

* hyperparameter $\beta$
* hyperparameter $\lambda$
* `__pairs_of_B_indices__` = order of permutations of subscripts order of the second term on the RHS of eqn. (28), 6 $\times$ 4 in dimension 
* `__pairs_of_C_indices__` = order of permutations of subscripts order of the third term on the RHS of eqn. (28)  3 $\times$ 4 in dimension


## Notes
* $\gamma_2$, unlike $\kappa$ and $\gamma_1$ does not have any pair of repeated
 indices, e.g. 1122, nor 2211 nor 1111 etc., so
 for small angular separation, only $\kappa$ and $\gamma_1$ has increased
covariances on the diagonal compared to $\psi_s$  

## Parameters 
The variable that the ExpSquaredKernel and the DerivativeKernel uses is $l^2 =
1 / \beta$.

## Thoughts on implementation 
* The metric object should incorporate the $\delta_{ij}$ condition for
 diagonal D, which will kill a lot of terms (sorry for being pedantic
about including $\delta$ since I don't want myself to forget about it)

## Comparison between parametrization of George and our parametrization
## Test 1: 
Let's check that our general expression of the 4th derivative of $\Sigma$
is correct by working out an example 

