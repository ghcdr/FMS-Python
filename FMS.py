from Scene import Scene
from Graphics import Renderer
from Solver import Solver
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", metavar="SIM", default="chain", help="Simulation scene")
    parser.add_argument("--draw", metavar="DRAW", default="wireframe", help="Draw type")
    args = parser.parse_args()
    try:
        newScene = Scene()
        if args.sim not in newScene.sceneOpt:
            raise ValueError("Not a valid constructor.")
        newScene.sceneOpt[args.sim]()
        param = newScene.getState0()
        solver = Solver(param['positions'], param['masses'], param['springs'], param['fixed'], 'Newton')
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