from Solver import vector3, spring, Solver
import numpy as np


def makeChain(p=vector3(-4,0,0), v=vector3(5,0,0), n=3, k=0.7):
    "Chain constructor"
    positions = []
    masses = []
    springs = []
    for i in range(0, n):
        positions.append(p + i * v)
        if i == 0 or i == n - 1: 
            masses.append(0.9)
        else: 
            masses.append(0.9)
        if i < n - 1: 
            springs.append(spring(i, i + 1, k, np.linalg.norm(v)))
        else: pass
    return { 'positions': positions, 'masses': masses, 'springs': springs }


def makeCloth(p=vector3(-4,0,0), v=vector3(5,0,0), width=5, height=5, k=0.9, mass=0.05):
    "Cloth constructor"
    positions = []
    masses = []
    springs = []
    skip = np.linalg.norm(p) * vector3(0, 1, 0)
    for i in range(0, height):
        for j in range(0, width):
            positions.append(p + i * v + j * skip)
            masses.append(mass)
            if j > 0:
                springs.append(spring(i*width + j - 1, i*width + j, k, np.linalg.norm(v)))
            if i > 0:
                springs.append(spring(i*width + j - width, i*width + j, k, np.linalg.norm(v)))
            if i > 0 and j > 0:
                springs.append(spring(i*width + j - 1 - width, i*width + j, k, np.linalg.norm(v)))
    return { 'positions': positions, 'masses': masses, 'springs': springs }


# 'Scene' constructors
Scene = {
    'chain': makeChain,
    'cloth': makeCloth
}