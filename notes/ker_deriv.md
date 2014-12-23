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
\vec{x_2})_i$ or $g_i = (\vec{y_1}, \vec{y_2})$  where we have data points. 
As usual, first column,
$x_1$ or $y_1$ is the first spatial
dimension, $x_2$ or $y_2$ is the second one, the i-th row correspond to spatial
coordinates of the i-th data point.  
$\vec{x}$ and $\vec{y}$ are the same but we call them different names for
denoting their location in the covariance matrix ....

For inferring the convergence and shear, we need the 2nd spatial
derivatives.
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
correspond to the spatial dimensions
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
r^2 = (t - t')^T D (t - t') 
\end{equation}

Since we are working in projected (2D) space, D is a 2 $\times$ 2 matrix.
More explicitly, in the GP model:
\begin{align*}
r^2_{ij} & = (x_i - x_j, y_i - y_j)
\left(
\begin{array}{cc}
D_{11} & D_{12} \\ D_{21} & D_{22} 
\end{array}
\right)
\left(
\begin{array}{c}
x_i - x_j \\ y_i - y_j  
\end{array}
\right)
\\
\Sigma_{ij} &= \lambda^{-1} \exp\left( -\frac{\beta}{2} r^2_{ij} \right)
\end{align*}

An example of $r^2$ with an Euclidean metric for a pair of data points, $t_i, t_j$
would be:
\begin{equation*}
r^2 = (x_i-x_j)^2 + (y_i-y_j)^2 
\end{equation*}

# Summary: basic derivatives with the preceeding coefficients



## Comparison between parametrization of George and our parametrization



