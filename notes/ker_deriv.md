---
title: "Notes - A GP model for shear fields"
author: "Karen Y. Ng"
date: "December 19, 2014"
output: ioslides_presentation
---

# Our Gaussian process "model" for the  projected lensing potential 
\begin{align}
\psi(\vec{x}, \vec{y}) \sim N(0, \Sigma(\vec{x}, \vec{y}))
\end{align}
which is a scalar field evaluated at the positions $(\vec{x}, \vec{y})$ where we have data
points

For inferring the convergence and shear, we need the 2nd spatial
derivatives.
The subscripts in these WL equations correspond to the spatial
coordinates $x, y$ instead of the observation numbers 
i.e. i, j = 1, 2, ..., n observations
\begin{align*}
\kappa &= \frac{1}{2}tr(\psi_{,ij})\\ 
&= \frac{1}{2} (\psi_{,11} + \psi_{,22})\\ 
&=
\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x^2} +
\frac{\partial^2 \psi}{\partial y^2 }\right)
\end{align*}

\begin{align*}
\gamma_1 &= \frac{1}{2} (\psi_{,11} - \psi_{,22})\\ 
&=\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x^2} - 
\frac{\partial^2 \psi}{\partial y^2}\right)
\end{align*}

\begin{align*}
\gamma_2 &= \frac{1}{2} (\psi_{,12} + \psi_{,21})\\ 
&=\frac{1}{2}\left(\frac{\partial^2 \psi}{\partial x \partial y} + 
\frac{\partial^2 \psi}{\partial y \partial x}\right)
\end{align*}

# Covariances of the required functions 
Note that $\psi, \kappa$ and $\gamma$ are scalar fields. 
However, we are evaluating them at the locations of the data points
$(x_i, y_i)$, therefore, when we are writing down the shorthand for the
i, j subscripts below, we mean, we first take the spatial derivatives of
those scalar field(s) with
respect to x or y, then evaluate them at $(x_i, y_i)$.
The spatial derivatives are represented as follows: 
\begin{equation*}
\psi_{,1} = \frac{\partial \psi}{\partial x} 
\end{equation*}
etc. with a comma in the subscript. 

Also note expectation and derivative are both linear operators, so we can
exchange their positions (and try not to let mathemticians read this and
shoot us)
\begin{align*}
Cov_{ij}(\kappa) &= \mathbb{E}\left[ 
(\kappa - \mathbb{E}[\kappa])|_i 
(\kappa - \mathbb{E}[\kappa])|_j 
\right]\\
&= \mathbb{E}\left[ 
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
\right) \psi 
\right]
\right]\bigg|_i
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
\right) \psi 
\right]
\right]\bigg|_j
\right]\\
&=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x^2} +
\frac{\partial^2}{\partial y^2}
\right)
[\psi - \mathbb{E}[\psi]]|_i
\left(
\frac{\partial^2}{\partial x^2} +
\frac{\partial^2}{\partial y^2}
\right)
[\psi - \mathbb{E}[\psi]]|_j
\right]\\
&= \frac{1}{4}\left(
\frac{\partial^4}{\partial x^4} + 
\frac{\partial^2}{\partial x^2} \frac{\partial^2}{\partial y^2} +
\frac{\partial^2}{\partial y^2} \frac{\partial^2}{\partial x^2} + 
\frac{\partial^4}{\partial y^4}  
\right) \Sigma_{ij} \\
\end{align*}
\begin{align}
Cov(\kappa)&= \frac{1}{4}\left(
\Sigma_{,1111} + \Sigma_{,1122} + \Sigma_{,2211} + \Sigma_{,2222}
\right)
\end{align}

Similarly,

\begin{align*}
Cov_{ij}(\gamma_1) &= \mathbb{E}\left[ 
(\gamma_1- \mathbb{E}[\gamma_1])|_i 
(\gamma_1- \mathbb{E}[\gamma_1])|_j 
\right]\\
&= \mathbb{E}\left[ 
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}
\right) \psi 
\right]
\right]\bigg|_i
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}
\right) \psi 
\right]
\right]\bigg|_j
\right]\\
&=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x^2} - 
\frac{\partial^2}{\partial y^2}
\right)
[\psi - \mathbb{E}[\psi]]|_i
\left(
\frac{\partial^2}{\partial x^2} - 
\frac{\partial^2}{\partial y^2}
\right)
[\psi - \mathbb{E}[\psi]]|_j
\right]\\
&= \frac{1}{4}\left(
\frac{\partial^4}{\partial x^4} - 
\frac{\partial^2}{\partial x^2} \frac{\partial^2}{\partial y^2} -  
\frac{\partial^2}{\partial y^2} \frac{\partial^2}{\partial x^2} + 
\frac{\partial^4}{\partial y^4}  
\right) \Sigma_{ij} \\
\end{align*}

\begin{align}
Cov(\gamma_1)&= \frac{1}{4}\left(
\Sigma_{,1111} - \Sigma_{,1122} - \Sigma_{,2211} + \Sigma_{,2222}
\right)
\end{align}

And, 

\begin{align*}
&Cov_{ij}(\gamma_2) \\
&= \mathbb{E}\left[ 
(\gamma_2- \mathbb{E}[\gamma_2])|_i 
(\gamma_2- \mathbb{E}[\gamma_2])|_j 
\right]\\
&= \mathbb{E}\left[ 
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x \partial y} - \frac{\partial^2}{\partial y
\partial x}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x \partial y} - \frac{\partial^2}{\partial y
\partial x}
\right) \psi 
\right]
\right]\bigg|_i
\left[
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x \partial y} - \frac{\partial^2}{\partial y
\partial x}
\right) \psi -
\mathbb{E}\left[ 
\frac{1}{2} 
\left( 
\frac{\partial^2}{\partial x \partial y} - \frac{\partial^2}{\partial y
\partial x}
\right) \psi 
\right]
\right]\bigg|_j
\right]\\
&=\frac{1}{4}\mathbb{E}\left[
\left(
\frac{\partial^2}{\partial x \partial y} - 
\frac{\partial^2}{\partial y \partial x}
\right)
[\psi - \mathbb{E}[\psi]]|_i
\left(
\frac{\partial^2}{\partial x \partial y} - 
\frac{\partial^2}{\partial y \partial x}
\right)
[\psi - \mathbb{E}[\psi]]|_j
\right]
\end{align*}

\begin{align}
Cov(\gamma_2)&= \frac{1}{4}\left(
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

# Summary: basic derivatives with the preceeding coefficients
\begin{align}
\frac{\partial r^2}{\partial x_i} &= [(D + D^T) (\vec{x} - \vec{y})]_i\\
\frac{\partial r^2}{\partial y_i} &= - [(D + D^T) (\vec{x} - \vec{y})]_i \\
{\rm Hess}(r^2(\vec{x}, \vec{y})) &= (D + D^T) \left(
					\begin{array}{cc}
					1 & -1 \\
					-1 & 1 \\  
					\end{array}
					\right) 
					\end{align}



## Comparison between parametrization of George and our parametrization



