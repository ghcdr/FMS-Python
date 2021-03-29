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

    def makeChain(self, p=vector3(-8,0,0), v=vector3(4,0,0), n=5, k=0.9, mass=0.05):
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

    def makeCloth(self, p=vector3(0,0,0), v=vector3(5,0,0), width=5, height=5, k=0.9, mass=0.05):
        """
        Cloth constructor.
        P: first corner
        v: direction 
        width: width in particles
        height: height .
        k: all springs' stiffness
        mass: all particles' mass
        """
        positions = self.state0['positions']
        masses = self.state0['masses']
        springs = self.state0['springs']
        fixed = self.state0['fixed']
        indices = []
        obj = Object(2, len(self.state0['masses']), len(self.state0['masses']) + width * height)
        skip = np.linalg.norm(p) * vector3(0, -1, 0)
        for i in range(0, height):
            for j in range(0, width):
                positions.append(p + j * v + i * skip)
                masses.append(mass)
                if j > 0:
                    springs.append(spring(i*width + j - 1, i*width + j, k, np.linalg.norm(v)))
                if i > 0:
                    springs.append(spring(i*width + j - width, i*width + j, k, np.linalg.norm(v)))
                if i > 0 and j > 0:
                    springs.append(spring(i*width + j - 1 - width, i*width + j, k, np.linalg.norm(v)))
                if i < height - 1 and j < width - 1:
                    obj.addIdx((i*width + j, i*width + j + 1, i*width + j + width))
                    obj.addIdx((i*width + j + 1, i*width + j + width, i*width + j + width + 1))
        # Fix corners
        fixed.add(len(springs))
        springs.append(spring(0, 0, k, 0))
        fixed.add(len(springs))
        springs.append(spring(width - 1, width - 1, k, 0))
        # Store object
        self.objects.append(obj)