from panda3d.core import MeshDrawer
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import *
from random import *
import sys


class Renderer(ShowBase):
    "Specialized renderer for mass-spring systems"
    def __init__(self, simulator, primitive = 'lines'):
        ShowBase.__init__(self)
        self.primitive = primitive
        self.sim = simulator
        self.mesh = MeshDrawer()
        # Keys
        self.accept("escape", sys.exit)
        # General setup
        base.cam.setPos(2.5, -30, 0)
        r = self.mesh.getRoot()
        r.reparentTo(render)
        r.setDepthWrite(False)
        r.setTransparency(True)
        r.setTwoSided(True)
        r.setBin("fixed", 0)
        r.setLightOff(True)
        # Register drawing callback
        if primitive == 'lines': taskMgr.add(self.wireframe_draw, "sim_draw")
        # Enable simulation stepping
        self.dt = 0.0
        #taskMgr.add(self.updateSim, "sim_update")

    def updateSim(self, task):
        "Task for stepping the simulator forward"
        self.dt += globalClock.getFrameTime()
        if(self.dt >= self.sim.dt):
            self.sim.step()
            self.dt = 0.0
        return task.cont # signals the task should be called over again

    def wireframe_draw(self, task):
        "Task for drawing the wireframe"
        self.mesh.begin(base.cam, render)
        for p in self.sim.getParticles(Vec3):
            self.mesh.particle(Vec3(p), Vec4(random(),random(),random(),1), .2, Vec4(0.9,0,0.1,1), 0)
        for s in self.sim.getSprings(Vec3):
            self.mesh.segment(Vec3(s[0]), Vec3(s[1]), Vec4(random(),random(),random(),1), .1,  Vec4(0,0,0.0,1))
        return task.cont # signals the task should be called over again