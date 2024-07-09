# TTStateMY

Algorithms and numerical examples for the paper
[Smoothed Moreau-Yosida Tensor Train Approximation of State-constrained Optimization Problems under Uncertainty](https://arxiv.org/abs/2301.08684) by Harbir Antil, Sergey Dolgov and Akwum Onwunta.

## Installation

These codes require MATLAB (any version above 2016 should be fine) and [TT-Toolbox](https://github.com/oseledets/TT-Toolbox).
Numerical experiment scripts (starting with `test_`) will try to download the TT-Toolbox and add it to the path automatically using a `check_tt` function. However, if this procedure fails, you need to download and setup the TT-Toolbox and certain quadrature rules yourself before running these codes.

## Numerical tests

- `test_1d_elliptic.m` One-dimensional elliptic PDE example. Fast enough for playing around.
- `test_2d_elliptic.m` 2D elliptic PDE.  _Slow._
- `test_vi.m` 2D elliptic variational inequality. Medium-fast.
- `test_SEIR_constrained.m` random SEIR model with contrained R number. _Slow_.

Each script will ask interactively to enter certain parameters (grid size, stopping tolerance, etc.). Default values are advised to reproduce the experiments in the paper and/or reduce the running time. In particular, `test_1d_elliptic.m` reproduces the experiment with $\gamma=300$ by default for faster computations. You may change the parameters as in the paper to reproduce the other plots.

## Algorithms

`test_vi.m` and `test_SEIR_constrained.m` are self-contained: the file starts with the actual experiment script, followed by functions it uses. Elliptic PDEs use precisely Algorithm 4.1 from the paper implemented in

- `moreau_yosida_fixpoint.m`

which depends on two generic functions:

- `cost_fun.m` computes the cost function
- `cost_grad.m` computes the gradient of the cost function

These three functions take the model parameters, but also function handles for computing the state solution, its gradient, as well as specific cost function and its gradient for the particular model. It can also take a function to plot the solutions at intermediate iterations. See **Elliptic PDE** sections below.

The `moreau_yosida_fixpoint.m` function outputs the following variables:

- `u` final control
- `y_tt` final state
- `ttranks_y` vector of max TT ranks of y in all iterations
- `ttranks_grad_MY` vector of max TT ranks of grad of MY term in all iterations
- `ttimes_y` vector of CPU times of solving for y in all iterations
- `ttimes_cost` vector of CPU times of computing cost in all iterations
- `ttimes_grad` vector of CPU times of computing gradient in all iterations

## Model solvers and utilities

#### 1D Elliptic PDE

- `solve_fun_elliptic_1d.m` Computes the forward solution (state)
- `grad_y_fun_elliptic_1d.m` Gradient of state over control
- `j_grad_fun_elliptic_1d.m` State cost, gradient or Hessian, including MatVec application
- `grad_my_elliptic_1d.m` Gradient of the Moreau-Yosida term over control
- `plot_elliptic_1d` Plot state, constraint and control.

#### 2D Elliptic PDE

- `solve_fun_elliptic_2d.m` Computes the forward solution (state)
- `grad_y_fun_elliptic_2d.m` Gradient of state over control
- `j_grad_fun_elliptic_2d.m` State cost, gradient or Hessian, including MatVec application
- `grad_my_elliptic_2d.m` Gradient of the Moreau-Yosida term over control
- `plot_elliptic_2d` Plot state, constraint and control.

#### Variational Inequality

- `solve_y_vi.m` Computes the forward solution (state)
- `state_cost_vi.m` Computes the cost
- `solve_adj_vi.m` Computes the adjoint state and the gradient of the cost
- `mean_field_hess_vi.m` Computes the Hessian of the cost at the mean fixed point

#### SEIR

- `SEIRcost.m` A function to compute the cost function of SEIR model on given samples
- `SEIRFDGrad.m` Finite Difference approximation of the gradient of the SEIR cost.
- `SEIRIEuler.m` Implicit Euler method to solve the forward SEIR ODE problem.
- `plot_prior_Ic.m` Plot the histogram and confidence interval for the controlled $I_C$ state.
- `SEIRData/` [Historic England data](https://doi.org/10.1371/journal.pcbi.1009236) for SEIR model.

## Utilities

#### Quadratures in random variables

- `check_quadratures.m` will check and if necessary download the quadrature rule from the MATLAB repository:
- `lgwt.m`  [Legendre-Gauss Quadrature Weights and Nodes](https://uk.mathworks.com/matlabcentral/fileexchange/4540-legendre-gauss-quadrature-weights-and-nodes).

If the automatic download fails, you need to download `lgwt.m` yourself before using the other codes.

#### Interpolation functions

- `lagrange_interpolant.m` A function to interpolate a Lagrangian polynomial on any grid.
- `cheb2_interpolant.m`  Barycentric interpolation from the Chebyshev-2 grid.
- `tt_sample_lagr.m`  Interpolates a TT decomposition via tensor product Lagrange polynomials on the Chebyshev grid.

#### Smoothing functions

- `logsmooth.m`  Log-sigmoid smoothing function, applied to data elementwise.
- `grad_logsmooth.m` First derivative of Log-sigmoid.

#### TT approximation

- `amen_cross_s.m`   Enhanced TT-Cross algorithm for the TT approximation.

#### Parsing

- `parse_parameter.m` A function to read parameters from the keyboard.
- `check_tt.m` Check/download/add-to-path for TT-Toolbox.

## Further docs

Each function file contains its own description in the first comment. See e.g.

```matlab
help('moreau_yosida_fixpoint')
```

or open the file in the editor.
