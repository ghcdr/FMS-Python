import numpy as np
from copy import copy
from scipy.sparse.csc import csc_matrix
from scipy.sparse import kron, identity, diags
from scipy.sparse.linalg import factorized
from scipy.linalg import cholesky, cho_factor, cho_solve, solve


spring_type = [('p1', int), ('p2', int), ('k', float), ('r', int)]

def vector3(e1, e2, e3):
    return np.array([e1, e2, e3])

def spring(p1, p2, k, r):
    return np.array((p1, p2, k, r), dtype=spring_type)


class Solver:
    "Implementation of the Fast Mass-Springs method (Liu et al.)"
    ndim    = 3     # 3-dimensional space
    m       = 0     # Particle Count
    s       = 0     # Spring count
    d       = None  # Stored spring orientations
    q       = None  # Positional simulation state
    q0      = None  # Previous positional simulation state
    M       = None  # Mass (diagonal) matrix 
    Minv    = None  # Inverse mass (diagonal) matrix 
    J       = None  # Some other matrix (eq. 12)
    L       = None  # The Laplacian
    A       = None  # 'A' as in 'Ax=b', in the problem of global step
    Ma      = None  # Inertial term
    LS      = None  # Linear system to solve
    Ch      = None  # Cholesky factorization of 'A'
    b       = None  # 'b' in the same context 
    springs = None  # Spring storage as a list of 'spring_type' (defined above)
    fext    = None  # Per particle external forces (accelerations, at first, converted then to forces).
    g       = None  # Acceleration of gravity, which is also converted to force afterwards 
    dt      = 1.0 / 60.0 # Timestep 
    dt2     = dt ** 2.0

    def __init__(self, positions, masses, springs):
        # General state setup
        self.g = vector3(0.0, -9.8, 0.0)
        self.m = len(positions)
        self.s = len(springs)
        self.q0 = np.array(positions).reshape(self.ndim * self.m, 1)
        self.q = copy(self.q0)
        self.M = kron(diags(masses), np.eye(self.ndim), format='csc')
        self.Minv = kron(diags(list(map(lambda x : 0 if x==0 else 1.0/x, masses))), np.eye(self.ndim), format='csc')
        self.springs = np.array(springs, dtype=spring_type)
        self.d = np.empty((self.ndim * self.s, 1))
        # Matrices L and J setup
        self.L = csc_matrix((self.m, self.m))
        self.J = csc_matrix((self.m, self.s))
        for idx, s in enumerate(self.springs):
            Ai = csc_matrix(([1, -1],([s['p1'], s['p2']], [0, 0])), shape=(self.m, 1))
            self.L += s['k'] * Ai * Ai.transpose()
            self.J += s['k'] * Ai * csc_matrix(([1.0], ([idx], [0])), shape=(self.s, 1)).transpose()
        self.L = kron(self.L, np.eye(self.ndim), format='csc')
        self.J = kron(self.J, np.eye(self.ndim), format='csc')
        # Matrix A precomputation (Global step)
        self.A = self.M + self.dt2 * self.L
        #self.LS = factorized(self.A)
        try:
            self.Ch = cho_factor(self.A.toarray())
        except LinAlgError as e:
            print("Cholesky failed: ", e)
        # Initialize external forces
        self.fext = np.zeros((self.ndim * self.m, 1))

    def __del__(self):
        pass

    def __local(self):
        "Local step: minimization of eq. 14 w.r.t. d"
        positions = self.q.reshape(self.m, self.ndim)
        for i, s in enumerate(self.springs):
            d = positions[s['p1']] - positions[s['p2']]
            length = np.linalg.norm(d)
            d = s['r'] * (vector3(0, 0, 0) if length == 0.0 else (d / length))
            self.d.reshape(self.s, self.ndim)[i] = d

    def __global(self):
        "Global step: minimization of eq. 14 w.r.t. x"
        self.b = self.dt2 * self.J * self.d + self.Ma + self.dt2 * self.fext
        #self.q = self.LS(self.b)
        # Using Cholesky
        self.q = cho_solve(self.Ch, self.b)
        # Manually fixing cloth corners
        self.q.reshape(self.m, self.ndim)[24] = self.q0.reshape(self.m, self.ndim)[24]
        self.q.reshape(self.m, self.ndim)[4] = self.q0.reshape(self.m, self.ndim)[4]
        
    def getParticles(self):
        "Returns particles' positions"
        return self.q.reshape(self.m, self.ndim)

    def set_dt(self, val):
        "Update dt, dt2, and recompute A = M + h^2L"
        self.dt = val
        self.dt2 = val ** 2.0
        self.A = self.M + self.dt2 * self.L

    def apply_forces(self, acc):
        "Actually, acceleration"
        self.fext = np.array(acc).reshape(self.ndim * self.m, 1)

    def step(self, max_iterations=5, max_error=0.001):
        "Advance simulation by one step"
        # Compute resulting external forces
        self.fext = self.M * (self.fext.reshape(self.m, self.ndim) + self.g).reshape(self.ndim * self.m, 1)
        # The inertial term
        self.Ma = self.M * (2.0 * self.q - self.q0)
        self.q0 = copy(self.q)
        # Solve implicit integration
        for _ in range(0, max_iterations):
            self.solve()

    def solve(self):
        "Alternate between local and global steps"
        self.__local()
        self.__global()
        # TODO: return error