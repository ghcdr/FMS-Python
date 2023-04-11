from Scene import Scene
from Graphics import Renderer
from Solver import Solver
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", metavar="SIM", default="chain", help="Simulation scene")
    #parser.add_argument("--dim", metavar="DIM", default=5, help="Object's dimensions")
    parser.add_argument("--prof", metavar="PRF", default=0, help="Step rate at which the simulation will be profiled")
    parser.add_argument("--method", metavar="MTH", default="A2FOM", help="Method", choices=['FMS', 'Newton', 'Jacobi'])
    args = parser.parse_args()
    try:
        newScene = Scene()
        if args.sim not in newScene.sceneOpt:
            raise ValueError("Not a valid constructor.")
        newScene.sceneOpt[args.sim]()
        param = newScene.getState0()
        solver = Solver(param['positions'], param['masses'], param['springs'], param['fixed'], args.method, args.prof)
        app = Renderer(solver, newScene.getObjects(), mTd=False)
        app.run()
    except ValueError as e:
        print(e, "Valid options are: ")
        print("interactive", *newScene.sceneOpt, sep=", ")
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1) 
    except: 
        # exiting from panda3d main, as it always throws
        sys.exit(0)
     
if __name__ == '__main__':
    main()