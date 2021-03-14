from Scene import Scene
from Graphics import Renderer
from Solver import Solver
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", metavar="SIM", default="cloth", help="Simulation scene")
    parser.add_argument("--draw", metavar="DRAW", default="wireframe", help="Draw type")
    args = parser.parse_args()
    try:
        if(args.sim not in Scene):
            raise ValueError("Not a valid simulation")
        param = Scene[args.sim]()
        solver = Solver(param['positions'], param['masses'], param['springs'])
        app = Renderer(solver)
        app.run()
    except Exception as e:
        print(e)
        sys.exit(1)
    except: 
        # exiting from panda3d main, as it always throws
        sys.exit(0)
       
if __name__ == '__main__':
    main()