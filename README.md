# Model-Based Control using Koopman Operators

The Koopman operator $\mathcal{K}$ is an infinite dimensional linear operator that directly acts on the functions of state 
$$
    \mathcal{K}g = g \circ F,
$$
where $\circ$ is the composition operator such that
$$
    \mathcal{K}g(x(t_i)) = g(F(x(t_i))) = g(x(t_{i+1})).
$$
That is, the Koopman operator $\mathcal{K}$ takes *any* observation of state $g(x(t_i))$ at time $t_i$ and evolves the function of state subject to its dynamics forward in time *linearly*. Thus, a nonlinear dynamical system can be represented by a linear Koopman operator in a lifted function space where the observations of state evolve linearly. The contribution in robotics is that now a highly nonlinear robotic system can be represented as a linear dynamical system which contains all the nonlinear information (as opposed to a Taylor expansion of the dynamics centered at an equilibrium state).
