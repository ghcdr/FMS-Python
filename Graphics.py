from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from random import *
import sys


class Renderer(ShowBase):
    "Specialized renderer for mass-spring systems"

    dt = 0.0   # Frame delta time
    sim = None  # Simulator instance
    mesh = MeshDrawer() # MeshDrawer

    def __init__(self, simulator):
        ShowBase.__init__(self)
        props = WindowProperties()
        props.setTitle('FMS')
        base.win.requestProperties(props)
        self.sim = simulator
        # Keys
        self.accept("escape", sys.exit)
        # Reposition camera, with -z as front
        base.useDrive()
        base.useTrackball()
        base.cam.setPos(0, 0, 0)
        base.cam.setHpr(0, -90, 0)
        base.cam.setPos(0, 0, 80)
        # General setup
        self.setBackgroundColor((1, 1, 1, 1))
        r = self.mesh.getRoot()
        #r.setTexture(loader.loadTexture(".png"))
        r.reparentTo(render)
        #r.setDepthWrite(False)
        r.setTransparency(True)
        r.setTwoSided(True)
        r.setBin("fixed", 0)
        r.setLightOff(True)
        # Register drawing callback
        taskMgr.add(self.wireframe_draw, "sim_draw")
        # Enable simulation stepping
        taskMgr.add(self.updateSim, "sim_update")

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
        #for p in self.sim.getParticles(Vec3):
        #    self.mesh.particle(Vec3(p), Vec4(random(),random(),random(),1), .7, Vec4(0.7,0,0.9,1), 0)
        for s in self.sim.getSprings(Vec3):
            self.mesh.segment(Vec3(s[0]), Vec3(s[1]), Vec4(random(),random(),random(),1), .1,  Vec4(0.0,0.0,0.0,1))
        # special case draw
        p  = self.sim.getParticles(Vec3)
        width = 5
        for a in range(0, 4):
            for b in range(0, 4):
                i = a * width + b
                self.mesh.tri(p[i], Vec4(0.7,0,0.9,1), Vec2(0,0), p[i + 1], Vec4(0.7,0,0.9,1), Vec2(0,0), p[i + width], Vec4(0.7,0,0.9,1), Vec2(0,0))
                self.mesh.tri(p[i + 1], Vec4(0.7,0,0.9,1), Vec2(0,0), p[i + width + 1], Vec4(0.7,0,0.9,1), Vec2(0,0), p[i + width],Vec4(0.7,0,0.9,1), Vec2(0,0))
        return task.cont # signals the task should be called over again