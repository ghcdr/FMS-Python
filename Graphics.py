from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
from random import *
import sys



class Renderer(ShowBase):
    """Specialized renderer for mass-spring systems"""
    sim = None  # Simulator instance
    mesh = MeshDrawer()
    objects = None
    particleSize = .7
    segmentSize = .2
    use_main_thread = True

    def __init__(self, simulator, objects, mTd):
        ShowBase.__init__(self)
        props = WindowProperties()
        props.setTitle('FMS')
        base.win.requestProperties(props)
        # State setup
        self.objects = objects
        self.sim = simulator
        # Keys
        self.accept("escape", sys.exit)
        # Reposition camera, with -z as front
        base.useDrive()
        base.useTrackball()
        base.cam.setPos(0, 0, 0)
        base.cam.setHpr(0, -90, 0)
        base.cam.setPos(0, 0, 100)
        # General setup
        self.setBackgroundColor((1, 1, 1, 1))
        r = self.mesh.getRoot()
        #r.setTexture(loader.loadTexture(".png"))
        r.reparentTo(render)
        r.setDepthWrite(False)
        r.setTransparency(True)
        r.setTwoSided(True)
        r.setBin("fixed", 0)
        r.setLightOff(True)
        # Register drawing callbacks
        taskMgr.add(self.draw, "draw")
        if self.use_main_thread:
            taskMgr.add(self.updateSimMain, "updateSimMain")
        else:
            # Since drawing is currently managed by tasks, we make physics run independently
            taskMgr.setupTaskChain('physics_chain', numThreads = 1, threadPriority=TP_low)
            taskMgr.add(self.updateSim, 'updateSim', taskChain = 'physics_chain')

    def draw(self, task):
        self.mesh.begin(base.cam, render)
        self.drawTriangles()
        self.drawParticles()
        self.drawSprings()
        self.mesh.end()
        return task.cont

    def updateSimMain(self, task):
        """Main thread: simply step once called"""
        self.sim.step()
        return task.cont

    def updateSim(self, task):
        """Separate thread: unfinished"""
        total = globalClock.getDt()
        if total >= self.sim.get_dt():
            # consume elapsed time
            while total > 0.0:
                beg = globalClock.getRealTime()
                self.sim.step()
                total -= max(self.sim.get_dt(), globalClock.getRealTime() - beg)
        return task.cont # signals the task should be called over again

    def drawTriangles(self):
        p = self.sim.getParticles()
        for obj in self.objects:
            if obj.dim != 2: continue
            for T in obj.indices:
                tcolor = Vec4(obj.tcolor)
                self.mesh.tri(Vec3(tuple(p[T[0]])), tcolor, Vec2(0, 0), Vec3(tuple(p[T[1]])), tcolor, Vec2(0, 0), Vec3(tuple(p[T[2]])), tcolor, Vec2(0, 0))
        
    def drawParticles(self):
        p = self.sim.getParticles()
        for obj in self.objects:
            pcolor = Vec4(obj.pcolor)
            for i in range(obj.beg, obj.end):  
                self.mesh.particle(Vec3(tuple(p[i])), Vec4(random(), random(), random(), 1), self.particleSize, pcolor, 0)

    def drawSprings(self):
        p = self.sim.getParticles()
        for obj in self.objects:
            for I in obj.indices:
                scolor = Vec4(obj.scolor)
                if obj.dim == 1:
                    self.mesh.segment(Vec3(tuple(p[I[0]])), Vec3(tuple(p[I[1]])), Vec4(random(), random(), random(), 1), self.segmentSize, scolor)
                if obj.dim == 2:
                    self.mesh.segment(Vec3(tuple(p[I[0]])), Vec3(tuple(p[I[1]])), Vec4(random(), random(), random(), 1), self.segmentSize, scolor)
                    self.mesh.segment(Vec3(tuple(p[I[1]])), Vec3(tuple(p[I[2]])), Vec4(random(), random(), random(), 1), self.segmentSize, scolor)
                    self.mesh.segment(Vec3(tuple(p[I[0]])), Vec3(tuple(p[I[2]])), Vec4(random(), random(), random(), 1), self.segmentSize, scolor)