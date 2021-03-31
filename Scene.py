from Solver import vector3, spring, Solver
import numpy as np
from random import *


class Object:
    "Particle object."

    beg = None # first particle 
    end = None # last .
    dim = None # object dimension
    # TODO
    scolor = None # spring color
    pcolor = None # particle color
    tcolor = None # triangle color
    indices = []
    
    def __init__(self, dim, ini, fin):
        """
        dim: object's dimension
        ini: first index (wrt the scene storage)
        fin: last index .
        """
        self.dim = dim
        self.beg = ini
        self.end = fin
        # Colors of segments, particles, and triangles
        self.scolor = (random(), random(), random(), 1.0)
        self.pcolor = (random(), random(), random(), 1.0)
        self.tcolor = (random(), random(), random(), 1.0)

    def addIdx(self, idx):
        if len(idx) != self.dim + 1:
            raise Error(str(self.dim + 1) + " indices expected")
        self.indices.append(idx)

    def getIndices(self):
        return self.indices


class Scene:
    """Encapsulates objects being simulated and drawn."""

    objects = []
    state0 = { 'positions': [], 'masses': [], 'springs': [], 'fixed': set() }
    # 'Scene' constructors
    sceneOpt = {}

    def __init__(self):
        self.sceneOpt = {
            'chain': self.makeChain,
            'cloth': self.makeCloth
        }

    def __del__(self):
        pass

    def getObjects(self):
        return self.objects

    def getState0(self):
        return self.state0

    def makeChain(self, p=vector3(-8,0,0), v=vector3(4,0,0), n=10, k=0.9, mass=0.005):
        """
        Chain constructor.
        p: beginning position
        v: direction
        n: length in particles
        k: all springs' stiffiness
        mass: all particles' mass
        """
        positions = self.state0['positions']
        masses = self.state0['masses']
        springs = self.state0['springs']
        fixed = self.state0['fixed']
        obj = Object(1, len(self.state0['masses']), len(self.state0['masses']) + n)
        indices = []
        for i in range(0, n):
            positions.append(p + i * v)
            masses.append(mass)
            if i < n - 1: 
                springs.append(spring(i, i + 1, k, np.linalg.norm(v)))
                obj.addIdx((i, i + 1))
        # Fix points
        fixed.add(len(springs))
        springs.append(spring(0, 0, 1, 0))
        fixed.add(len(springs))
        springs.append(spring(n - 1, n - 1, 1, 0))
        self.objects.append(obj)

    def makeCloth(self, p=vector3(0,0,0), v=vector3(3,0,0), width=6, height=6, k=0.09, total_mass=0.1):
        """
        Cloth constructor.
        P: first corner
        v: direction 
        width: width in particles
        height: height .
        k: all springs' stiffness
        mass: all particles' mass
        """
        mass = total_mass / (width * height)
        positions = self.state0['positions']
        masses = self.state0['masses']
        springs = self.state0['springs']
        fixed = self.state0['fixed']
        indices = []
        obj = Object(2, len(self.state0['masses']), len(self.state0['masses']) + width * height)
        skip = np.linalg.norm(v) * vector3(0, -1, 0)

        def create_spring(i, j, k, l):
            springs.append(spring(i*width + j, k*width + l, k, np.linalg.norm(positions[i*width + j] - positions[k*width + l])))

        def add_triangle(i, j, k, l, m, n):
            obj.addIdx((i*width + j, k*width + l, m*width + n))

        alt = False
        for i in range(0, height):   
            for j in range(0, width):
                positions.append(p + j * v + i * skip)
                masses.append(mass)
                if j > 0: create_spring(i, j - 1, i, j)
                if i > 0: create_spring(i - 1, j, i, j)
                if i > 0 and j > 0:
                    alt = not alt
                    if alt:
                        create_spring(i - 1, j - 1, i, j)
                        add_triangle(i - 1, j - 1, i, j, i, j - 1)
                        add_triangle(i - 1, j - 1, i, j, i - 1, j)
                    else: 
                        create_spring(i - 1, j, i, j - 1)
                        add_triangle(i - 1, j, i, j - 1, i, j)
                        add_triangle(i - 1, j, i, j - 1, i - 1, j - 1)
        # Fix corners
        fixed.add(len(springs))
        springs.append(spring(0, 0, 1, 0))
        fixed.add(len(springs))
        springs.append(spring(width - 1, width - 1, 1, 0))
        # Store object
        self.objects.append(obj)