import numpy as np
from pydrake.all import (
    Jacobian,
    MathematicalProgram,
    Solve,
    Variables,
    LinearQuadraticRegulator
)

g = 9.81
m = 1
l = 1
b = 0.1

# LQR design
x0 = [0, -1, 0]
u0 = [0]

A = np.array([
    [0, 1],
    [g * m * l / (m * l **2), -b / (m * l **2)]
])
B = np.array([
    [0],
    [1 / (m * l **2)]
])

Q = np.diag((10, 1))
R = [1e-2]

(K, S) = LinearQuadraticRegulator(A, B, Q, R)

prog = MathematicalProgram()
x = prog.NewIndeterminates(3, "x")
rho = prog.NewContinuousVariables(1, "rho")[0]

# Define the dynamics.
dx = [x[0] * x0[1] - x[1] * x0[0], x[2] - x0[2]]
u = u0 + K.dot(dx)

f = [
    x[1] * x[2],
    -x[0] * x[2],
    (-g * m * l * x[0] - b * x[2] + u[0])
]

# Define the Lyapunov function.
V = (S.dot(dx)).dot(dx)
Vdot = Jacobian([V], x).dot(f)[0]

# Define the Lagrange multiplier.
lambda_ = prog.NewFreePolynomial(Variables(x), 3).ToExpression()
mu_ = prog.NewFreePolynomial(Variables(x), 3).ToExpression()

prog.AddSosConstraint((V - rho) * x.dot(x) + mu_ * (x[0]**2 + x[1]**2) - lambda_ * Vdot)
prog.AddLinearCost(-rho)

result = Solve(prog)

assert result.is_success()

print(
    "Verified that "
    + str(V)
    + " < "
    + str(result.GetSolution(rho))
    + " is in the region of attraction."
)

assert np.fabs(result.GetSolution(rho) - 1) < 1e-5
