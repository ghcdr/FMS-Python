import numpy as np
from numpy.linalg import LinAlgError
from copy import copy
from scipy.sparse.csc import csc_matrix
from scipy.sparse import kron, diags
from scipy.sparse.linalg import factorized, inv
from scipy.linalg import cholesky, cho_factor, cho_solve, solve
from matplotlib import pyplot as plt


spring_type = [('p1', int), ('p2', int), ('k', float), ('r', int)]

def vector3(e1, e2, e3):
    return np.array([e1, e2, e3])

def spring(p1, p2, k, r):
    return np.array((p1, p2, k, r), dtype=spring_type)


class Solver:
    """Implementation of the Fast Mass-Springs method (Liu et al.)"""
    state0  = None  # Initial state
    method  = ''    # Integrator
    ndim    = 3     # 3-dimensional space
    m       = 0     # Particle Count
    s       = 0     # Spring count
    d       = None  # Stored spring orientations
    q       = None  # Positional simulation state
    q0      = None  # Previous positional simulation state
    qNext   = None  # State estimate for Newton's method
    M       = None  # Mass (diagonal) matrix 
    Minv    = None  # Inverse mass (diagonal) matrix 
    J       = None  # Some other matrix (eq. 12)
    L       = None  # The Laplacian
    A       = None  # 'A' as in 'Ax=b', in the problem of global step
    Ma      = None  # Inertial term
    Ch      = None  # Cholesky factorization of 'A'
    b       = None  # 'b' in the same context 
    springs = None  # Spring storage as a list of 'spring_type' (defined above)
    fixed   = None  # Fixed positions indices, stored as springs (pos, pos, k, 0)
    qFixed  = None  # The same as initial state; storage for fixed positions
    fext    = None  # Per particle external forces (accelerations, at first, converted then to forces).
    g       = None  # Acceleration of gravity, which is also converted to force afterwards 
    dt      = 1.0 / 60.0 # Timestep 
    dt2     = dt ** 2.0
    max_iterations  = 50
    max_error       = 1.0 ** -3
    elapsed_time    = 0.0
    step_counter    = 0


    def __init__(self, positions, masses, springs, fixed, method='FMS'):
        # General state setup
        self.method = method
        self.g = -9.8
        self.m = len(positions)
        self.s = len(springs)
        self.q0 = np.array(positions).reshape(self.ndim * self.m, 1)
        self.q = copy(self.q0)
        self.fixed = fixed
        self.qFixed = copy(self.q0)
        self.M = kron(diags(masses), np.eye(self.ndim), format='csc')
        self.Minv = kron(diags(list(map(lambda x : 0 if x==0 else 1.0/x, masses))), np.eye(self.ndim), format='csc')
        self.springs = np.array(springs, dtype=spring_type)
        self.d = np.empty((self.ndim * self.s, 1))
        # Matrices L and J setup
        self.L = csc_matrix((self.m, self.m))
        self.J = csc_matrix((self.m, self.s))
        for idx, s in enumerate(self.springs):
            Ai = None
            if idx in self.fixed:
                Ai = csc_matrix(([1],([s['p1']], [0])), shape=(self.m, 1))
            else:
                Ai = csc_matrix(([1, -1],([s['p1'], s['p2']], [0, 0])), shape=(self.m, 1))
            self.L += s['k'] * Ai * Ai.transpose()
            self.J += s['k'] * Ai * csc_matrix(([1.0], ([idx], [0])), shape=(self.s, 1)).transpose()
        self.L = kron(self.L, np.eye(self.ndim), format='csc')
        self.J = kron(self.J, np.eye(self.ndim), format='csc')
        # Matrix A precomputation (Global step)
        self.A = self.M + self.dt2 * self.L
        self.Ch = cho_factor(self.A.toarray())
        # Initialize external forces
        self.fext = self.M * np.full((self.ndim * self.m, 1), self.g)
        # Store initial state
        self.state0 = {'q0': self.q}

    def __del__(self):
        pass
        
    def getParticles(self):
        """Returns particles' positions"""
        return self.q.reshape(self.m, self.ndim)

    def get_dt(self):
        return self.dt

    def set_dt(self, val):
        """Update dt, dt2, and recompute A = M + h^2L"""
        self.dt = val
        self.dt2 = val ** 2.0
        self.A = self.M + self.dt2 * self.L
        raise Error('Not implemented')

    def apply_forces(self, acc):
        """Actually, acceleration"""
        self.acc = np.array(acc);
        raise Error('Not implemented')

    def reset(self):
        self.q = copy(state0['q0'])
        self.step_counter=0
        self.elapsed_time=0.0
        raise Error('Not implemented')

    def __update_fext(self):
        self.fext = self.M * (np.full((self.m * self.ndim, 1), self.g))

    def __update_inertial_term(self):
        self.Ma = self.M * (2.0 * self.q - self.q0)

    def step(self):
        """Advance simulation by one step"""
        self.step_counter+=1
        self.elapsed_time+=self.dt
        self.__update_fext()
        if self.method == 'FMS': 
            self.step_LocalGlobal()
        elif self.method == 'Newton':
            self.step_Newton()
        else:
            raise Error('No implemented method matches ', self.method)

    """
    Local-Global's method definitions
    """
    def step_LocalGlobal(self):
        self.__update_inertial_term()
        self.q0 = copy(self.q)
        # Solve implicit integration
        for k in range(0, self.max_iterations):
            self.LocalGlobalsolve()

    def LocalGlobalsolve(self):
        """Alternate between local and global"""
        self.__local()
        self.__global()

    def __local(self):
        """Local step: minimization of eq. 14 w.r.t. d"""
        positions = self.q.reshape(self.m, self.ndim)
        for i, s in enumerate(self.springs):
            if i in self.fixed:
                # Attachments
                self.d.reshape(self.s, self.ndim)[i] = self.qFixed.reshape(self.m, self.ndim)[s['p1']]
                continue
            d = positions[s['p1']] - positions[s['p2']]
            ld = np.linalg.norm(d)
            d = s['r'] * (vector3(0, 0, 0) if ld == 0.0 else (d / ld))
            self.d.reshape(self.s, self.ndim)[i] = d

    def __global(self):
        """Global step: minimization of eq. 14 w.r.t. x"""
        self.b = self.dt2 * self.J * self.d + self.Ma + self.dt2 * self.fext
        # Using Cholesky
        self.q = cho_solve(self.Ch, self.b)


    """
    Newton's method definitions
    """
    def step_Newton(self):
        self.qNext = self.q + (self.q - self.q0)*self.dt
        self.q0 = copy(self.q)
        self.q = copy(self.qNext)
        for k in range(0, self.max_iterations):
            self.__Newton()

    def __Jacobian(self):
        """
        Computes the constraints' Jacobian
        g'(x) = (x - y)M + Nabla E(x)
        alpha_i = Nabla_{pi1} E (p_1 = p_{i1}, p_2 = p_{i2}) 
            = (|p_1 - p_2| - r)*((p_1 - p_2)/|p_1 - p_2|)) + F_ext
        Nabla E_i = [alpha_i, -alpha_i]
        """
        dim = self.ndim
        J = np.zeros((self.m * dim, 1))
        positions = self.q.reshape(self.m, dim)
        for idx, s in enumerate(self.springs):
            # Attachments
            if idx in self.fixed:
                J.reshape(self.m, dim)[s['p1']] += s['k'] * (positions[s['p1']] - self.qFixed.reshape(self.m, dim)[s['p1']])
                continue
            i, j = s['p1'], s['p2']
            d = positions[s['p1']] - positions[s['p2']]
            ld = np.linalg.norm(d)
            k, r = s['k'], s['r']
            sgrad = k * (ld - r) * (d/ld)
            J.reshape(self.m, dim)[i] += sgrad
            J.reshape(self.m, dim)[j] -= sgrad
        J = self.M*(self.q - self.qNext) + self.dt2 * (J - self.fext)
        return J

    def __Hessian(self):
        """
        Computes the constraints' Hessian
        g''(x) = M + J(Nabla E(x))
        J_j(Nabla_{pi1^2} E (p_1 = p_{i1}, p_2 = p_{i2}) =
            = 1 - r/|p1 - p2| - r(p1- p2)^2/|p1 - p2|^3
            = 1 - r/|p1 - p2|(1 - (p1- p2)^2/|p1 - p2|^2)
        J_{ij} E = [[J(Nabla E_i), -J(Nabla E_i)],[-J(Nabla E_i), J(Nabla E_i)]]
        """
        dim = self.ndim
        positions = self.q.reshape(self.m, dim)
        H = np.zeros((self.m * dim, self.m * dim))
        for idx, s in enumerate(self.springs):
            # Attachments
            if idx in self.fixed:
                H[s['p1']*dim:s['p1']*dim+dim, s['p1']*dim:s['p1']*dim+dim] += s['k'] * np.eye(dim)
                continue
            i, j = s['p1'], s['p2']
            d = positions[s['p1']].reshape(1, dim) - positions[s['p2']].reshape(1, dim)
            d2 = d.transpose() * d
            ld = np.linalg.norm(d)
            ld2 = ld ** 2
            k, r = s['k'], s['r']
            spring_hessian = k * (np.eye(dim) - (r/ld) * (np.eye(dim) - (d2/ld2)))
            H[i*dim:i*dim+dim, i*dim:i*dim+dim] += spring_hessian
            H[j*dim:j*dim+dim, j*dim:j*dim+dim] += spring_hessian
            H[i*dim:i*dim+dim, j*dim:j*dim+dim] -= spring_hessian
            H[j*dim:j*dim+dim, i*dim:i*dim+dim] -= spring_hessian
        H = self.M + self.dt2 * H
        return H

    def __Newton(self):
        """One Newton iteration"""
        J = self.__Jacobian()
        H = self.__Hessian()
        Ch = cho_factor(H)
        step = cho_solve(Ch, J)
        self.q = self.q - step
