import numpy as np
from numpy.linalg import LinAlgError
from copy import copy
from scipy.sparse.csc import csc_matrix
from scipy.sparse import kron, diags, identity
from scipy.sparse.linalg import factorized, inv
from scipy.linalg import cholesky, cho_factor, cho_solve, solve
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import shift


spring_type = [('p1', int), ('p2', int), ('k', float), ('r', int)]

def vector3(e1, e2, e3):
    return np.array([e1, e2, e3])

def spring(p1, p2, k, r):
    return np.array((p1, p2, k, r), dtype=spring_type)


class Solver:
    """Implementation of the Fast Mass-Springs method (Liu et al.)"""
    state0              = None  # Initial state
    method              = ''    # Integrator
    ndim                = 3     # 3-dimensional space
    m                   = 0     # Particle Count
    s                   = 0     # Spring count
    d                   = None  # Stored spring orientations
    q                   = None  # Positional simulation state
    q0                  = None  # Previous positional simulation state
    inertia             = None  # Inertia
    M                   = None  # Mass (diagonal) matrix 
    Minv                = None  # Inverse mass (diagonal) matrix 
    J                   = None  # Some other matrix (eq. 12)
    L                   = None  # The Laplacian
    A                   = None  # 
    Ch                  = None  # 
    b                   = None  #  
    springs             = None  # 
    fixed               = None  # 
    qFixed              = None  # Storage for fixed positions
    fext                = None  # Per particle external forces
    g                   = None  # Acceleration of gravity
    dt                  = 1.0 / 30.0 # Timestep 
    dt2                 = dt ** 2.0
    max_iterations      = 20
    max_error           = 1.0 ** -20
    elapsed_time        = 0.0
    step_counter        = 0
    profiling_rate      = None
    implemented         = None

    m_iterations        = 6
    curr_anderson_it    = 0
    m_prev_G            = None
    m_prev_F            = None
    E_prev              = None
    qLast               = None
    lengths             = None
    gamma               = None

    def __init__(self, positions, masses, springs, fixed, method, profiling_rate=0):
        """
        method: 'Newton' | 'FMS' | 'Jacobi'
        profiling_rate: the step rate at which the performance will be graphed
        """
        # General state setup
        self.profiling_rate = profiling_rate
        self.method = method
        self.g = -9.8
        self.m = len(positions)
        self.s = len(springs)
        self.q0 = np.array(positions).reshape(self.ndim * self.m, 1)
        self.q = copy(self.q0)
        # Store initial state
        self.state0 = copy(self.q)
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
        # Implemented methods
        self.implemented = {
            'FMS':      self.step_LocalGlobal,
            'Jacobi':   self.step_Jacobi,
            'Newton':   self.step_Newton,
            'Anderson': self.step_Anderson,
            #'A2FOM':    self.step_A2FOM
        }

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
        self.q = copy(self.state0)
        self.q0 = copy(self.state0)
        self.step_counter = 0
        self.elapsed_time = 0.0
    
    def __update_internal_state(self):
        """Update control variables; profiling"""
        self.step_counter += 1
        self.elapsed_time += self.dt
        self.profile()

    def profile(self, triggered=False, all=False):
        all = True
        if (not triggered) and (self.profiling_rate == 0 or self.step_counter % self.profiling_rate != 0):
            return
        # Save state
        q0_store = copy(self.q0)
        q_store = copy(self.q)
        # X-axis values
        X = [*range(0, self.max_iterations)]
        # Y-axis
        for method in (self.implemented.keys() if all else [self.method]):
            Y = []
            self.__update_inertia()
            self.__update_fext()
            self.q0 = copy(self.q)
            self.q = copy(self.inertia)
            iterator = self.implemented[method]()
            for _ in range(0, self.max_iterations):
                iterator()
                Y.append(np.linalg.norm(self.__Gradient()))
            plt.plot(X, Y, label=method, linestyle='dashed')
            # Restore
            self.q0 = copy(q0_store)
            self.q = copy(q_store)
        # Plot   
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Step %d'% self.step_counter)
        plt.legend()
        plt.show()

    def __update_fext(self):
        self.fext = self.M.dot(
            np.full((self.m, self.ndim), np.array([0.0, self.g, 0.0])).reshape(self.m * self.ndim, 1))

    def __update_inertia(self):
        self.inertia = 2.0 * self.q - self.q0

    def __update_b(self):
        self.b = (self.dt2 * self.J * self.d) + (self.M * self.inertia) + (self.dt2 * self.fext)

    def total_energy(self):
        positions = self.q.reshape(self.m, self.ndim)
        spring_potentials = 0.0
        for s in self.springs:
            l = np.linalg.norm(positions[s['p1']] - positions[s['p2']]) - s['r']
            spring_potentials += .5 * s['k'] * l * l
        return .5 * (self.q - self.inertia).transpose().dot(self.M.dot(self.q - self.inertia))[0, 0] + \
            self.dt2 * (spring_potentials - self.q.T.dot(self.fext))[0, 0]

    def step(self):
        """Advance simulation by one step"""
        self.__update_internal_state()
        self.__update_inertia()
        self.__update_fext()
        self.q0 = copy(self.q)
        self.q = copy(self.inertia)
        if self.method not in self.implemented:
            raise Error('No implemented method matches ', self.method)
        iterator = self.implemented[self.method]()
        for _ in range(0, self.max_iterations):
            if iterator(): break          

    """
    Local-Global's method definitions
    """
    def step_LocalGlobal(self):
        return self.LocalGlobalsolve

    def LocalGlobalsolve(self):
        """Alternate between local and global"""
        self.__local()
        self.__global()
        return False

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
        self.__update_b() # update rhs
        self.q = cho_solve(self.Ch, self.b)

    """
    Newton's method definitions
    """
    def step_Newton(self):
        return self.__Newton

    def __springs_Jacobian(self):
        dim = self.ndim
        J = np.zeros((self.m * dim, 1))
        positions = self.q.reshape(self.m, dim)
        for idx, s in enumerate(self.springs):
            i, j = s['p1'], s['p2']
            k, r = s['k'], s['r']
            # Attachments
            if idx in self.fixed:
                J.reshape(self.m, dim)[i] += k * (positions[i] - self.qFixed.reshape(self.m, dim)[i])
                continue
            d = positions[i] - positions[j]
            ld = np.linalg.norm(d)
            sgrad = k * (ld - r) * (d/ld)
            J.reshape(self.m, dim)[i] += sgrad
            J.reshape(self.m, dim)[j] -= sgrad
        return J

    def __Gradient(self):
        """
        Computes the constraints' Jacobian
        g'(x) = (x - y)M + Nabla E(x)
        alpha_i = Nabla_{pi1} E (p_1 = p_{i1}, p_2 = p_{i2}) 
            = (|p_1 - p_2| - r)*((p_1 - p_2)/|p_1 - p_2|)) + F_ext
        Nabla E_i = [alpha_i, -alpha_i]
        """
        J = self.__springs_Jacobian()
        G = self.M.dot(self.q - self.inertia) + self.dt2 * (J - self.fext)
        return G

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
            i, j = s['p1'], s['p2']
            k, r = s['k'], s['r']
            # Attachments
            if idx in self.fixed:
                H[i*dim:i*dim+dim, i*dim:i*dim+dim] += k * np.eye(dim)
                continue
            d = positions[i].reshape(1, dim) - positions[j].reshape(1, dim)
            d2 = d.transpose() * d
            ld = np.linalg.norm(d)
            ld2 = ld ** 2.0
            spring_hessian = k * (np.eye(dim) - (r/ld) * (np.eye(dim) - (d2/ld2)))
            H[i*dim:i*dim+dim, i*dim:i*dim+dim] += spring_hessian
            H[j*dim:j*dim+dim, j*dim:j*dim+dim] += spring_hessian
            H[i*dim:i*dim+dim, j*dim:j*dim+dim] -= spring_hessian
            H[j*dim:j*dim+dim, i*dim:i*dim+dim] -= spring_hessian
        H = self.M + self.dt2 * H
        return H

    def __Newton(self):
        """One Newton iteration"""
        J = self.__Gradient()
        H = self.__Hessian()
        Ch = cho_factor(H)
        step = cho_solve(Ch, J)
        self.q = self.q - step
        return np.linalg.norm(J) < self.max_error

    """
    Jacobi
    """
    def step_Jacobi(self):
        return self.__Jacobi

    def __Jacobi(self):
        """Global step solved with Jacobi"""
        self.__local()
        # One Jacobi iteration
        self.__update_b()
        Minv = np.linalg.inv(np.diag(self.A.toarray())*np.eye(self.m * self.ndim))
        self.q = (np.eye(self.m * self.ndim) - Minv * self.A).dot(self.q) + Minv.dot(self.b)
        return False

    """
    Anderson Extrapolation
    """
    def step_Anderson(self):
        assert(self.m_iterations > 1)
        self.curr_anderson_it = 0
        self.col_idx = 0
        self.m_prev_G = np.zeros((self.ndim * self.m, self.m_iterations))
        self.m_prev_F = np.zeros((self.ndim * self.m, self.m_iterations))
        self.qLast = copy(self.q)
        self.E_prev = self.total_energy()
        self.lengths = np.ones(self.m_iterations)
        self.gamma = np.zeros((1))
        self.AA = np.zeros((self.m_iterations, self.m_iterations))

        return self.Anderson


    def Anderson(self):
        # Local-Global step, and residual computation
        #self.__local()
        # Check if Energy will decrease
        #E = self.total_energy()
        #if E >= self.E_prev:
        #    self.q = copy(self.qLast)
        #    self.__local()
        #self.E_prev = self.total_energy()
        F = copy(self.q)
        self.__local()
        self.__global()
        # Store latest result from Global step
        #self.qLast = copy(self.q)
        F = (self.q - F)[:, 0]

        if self.curr_anderson_it == 0:
            self.m_prev_G[:, 0] = -copy(self.q[:, 0])
            self.m_prev_F[:, 0] = -copy(F)

        else:
            p = copy(self.col_idx)

            self.m_prev_G[:, p] += self.q[:, 0]
            self.m_prev_F[:, p] += F

            self.lengths[p] = max(1e-14, np.linalg.norm(self.m_prev_F[:, p]))
            self.m_prev_F[:, p] /= self.lengths[p]

            mk = min(self.m_iterations, self.curr_anderson_it)

            if mk == 1:
                self.gamma[0] = np.array([0.0])
                Fnorm = np.linalg.norm(self.m_prev_F[: , p])
                self.AA[0, 0] = Fnorm ** 2.0
                if Fnorm > 1e-14:
                    self.gamma[0] = (self.m_prev_F[: , p] / Fnorm) @ (F / Fnorm)

            else:
                block = self.m_prev_F[:, p].T @ self.m_prev_F[:, 0:mk]
                block = block.reshape(mk)
                self.AA[p, 0:mk] = copy(block.T)
                self.AA[0:mk, p] = copy(block)
                Ch = cho_factor(self.AA[0:mk, 0:mk])
                self.gamma = cho_solve(Ch, self.m_prev_F[:, 0:mk].T @ F)

            self.q = self.q - (self.m_prev_G[:, 0:mk] @ (self.gamma / self.lengths[0:mk]).reshape(mk, 1))

            self.col_idx = (self.col_idx + 1) % self.m_iterations
            self.m_prev_F[:, self.col_idx] = -F
            self.m_prev_G[:, self.col_idx] = -self.q[:, 0]

        self.curr_anderson_it += 1

        return False
            

    def step_A2FOM(self):
        return self.A2FOM

    def A2FOM(self):

        pass
