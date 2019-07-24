# Model-Based Control using Koopman Operators
The Koopman operator is an infinite dimensional linear operator that directly acts on the functions of state. 
That is, the Koopman operator takes *any* observation of state at time t and evolves the functions of state subject to its dynamics forward in time *linearly*. Thus, a nonlinear dynamical system which has a discoverable Koopman operator can be represented by a linear Koopman operator in a lifted function space where the observations of state evolve linearly. 

The contribution in robotics is that now a highly nonlinear robotic system can be represented as a linear dynamical system which contains all the nonlinear information (as opposed to a Taylor expansion of the dynamics centered at an equilibrium state).
