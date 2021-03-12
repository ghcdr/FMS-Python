import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse import kron, identity, diags


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
        self.q0 = self.q = np.array(positions)
        self.M = kron(diags(masses), np.eye(self.ndim), format='csc')
        self.Minv = kron(diags(list(map(lambda x : 0 if x==0 else 1.0/x, masses))), np.eye(self.ndim), format='csc')
        self.springs = np.array(springs, dtype=spring_type)
        self.d = np.empty((self.q0.shape[1] * self.s, 1))
        # Matrices L and J setup
        self.L = csc_matrix((self.m, self.m))
        self.J = csc_matrix((self.m, self.s))
        for idx, s in enumerate(self.springs):
            Ai = csc_matrix(([1, -1],([s['p1'], s['p2']], [0, 0])), shape=(self.m, 1))
            self.L += s['k'] * Ai * Ai.transpose()
            self.J += s['k'] * Ai * csc_matrix(([1], ([idx], [0])), shape=(self.s, 1)).transpose()
        self.L = kron(self.L, np.eye(self.ndim), format='csc')
        self.J = kron(self.J, np.eye(self.ndim), format='csc')
        # Matrix A precomputation (Global step)
        self.A = self.M + self.dt2 * self.L
        # Initialize external forces
        self.fext = np.zeros((self.m, self.ndim))

    def __del__(self):
        pass

    def __local(self):
        "Local step: minimization of eq. 14 w.r.t. d"
        tmp = self.springs[:]['p1'] - self.springs[:]['p2']
        for d in tmp: d /= np.linalg.norm(d)
        self.d = tmp

    def __global(self):
        "Global step: minimization of eq. 14 w.r.t. x"
        
    def getParticles(self, const=None):
        "Returns particles' positions"
        if(const == None): return q
        else:
            l = []
            for x in self.q: l.append(const(tuple(x)))
            return l

    def getSprings(self, const=None):
        "Return springs' endpoints positions"
        l = []
        for s in self.springs:
            if const == None: l.append([self.q[s['p1']], self.q[s['p2']]])
            else: l.append([const(tuple(self.q[s['p1']])), const(tuple(self.q[s['p2']]))])
        return l

    def dt(self, val):
        "Update dt, dt2, and recompute A = M + h^2L"
        self.dt = val
        self.dt2 = val ** 2.0
        self.A = self.M + self.dt2 * self.L

    def apply_forces(self, acc):
        "Actually, acceleration"
        self.fext = acc

    def step(self, max_iterations=1, max_error=0.001):
        "Advance simulation by one step"
        # Compute resulting external forces
        self.fext = (self.fext + self.g) * self.M
        # Advance one time step
        self.b = self.dt2 * self.J *self.d + (self.q - self.q0) * self.M + self.dt2 * self.fext
        # Solve implicit integration
        for _ in range(0, max_iterations):
            self.solve()

    def solve(self):
        "Alternate between local and global steps"
        self.__local()
        self.__global()
        # TODO: return error