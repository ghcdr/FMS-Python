from Scene import Scene
from Graphics import Renderer
from Solver import Solver
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", metavar="SIM", default="chain", help="Simulation scene")
    args = parser.parse_args()
    try:
        if(args.sim not in Scene):
            raise ValueError("Not a valid simulation")
        param = Scene[args.sim]()
        solver = Solver(param['positions'], param['masses'], param['springs'])
        app = Renderer(solver, "lines")
        app.run()
    except Exception as e:
        #parser.print_help()
        print(e)
        sys.exit(1)
    except: # exiting from panda3d main
        sys.exit(0)
       
if __name__ == '__main__':
    main()