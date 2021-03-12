from Solver import vector3, spring, Solver
import numpy as np


def makeChain(p=vector3(0,0,0), v=vector3(2,0,0), n=4, k=0.7):
    positions = []
    masses = []
    springs = []
    for i in range(0, n):
        positions.append(p + i*v)
        if i == 0 or i == n - 1: masses.append(0.0)
        else: masses.append(1.0)
        if i < n - 1: springs.append(spring(i, i + 1, k, np.linalg.norm(v)))
        else: pass
    return { 'positions': positions, 'masses': masses, 'springs': springs }


# 'Scene' constructors
Scene = {
    'chain': makeChain
}