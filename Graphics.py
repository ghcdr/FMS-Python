from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import AmbientLight, DirectionalLight
from direct.task.Task import Task
from random import *
import sys



class Renderer(ShowBase):
    """Specialized renderer for mass-spring systems"""
    sim = None  # Simulator instance
    change_method = None
    sim_running = True
    stats = []
    mesh = MeshDrawer()
    objects = None
    particleSize = .6
    segmentSize = .2
    use_main_thread = True
    draw_wireframe = True
    draw_particles = True

    def __init__(self, simulator, objects, mTd):
        ShowBase.__init__(self)
        props = WindowProperties()
        props.setTitle('Demo')
        base.win.requestProperties(props)
        self.change_method = simulator.method
        self.title = OnscreenText(
            text=simulator.method,
            parent=base.a2dBottomRight, scale=.09,
            align=TextNode.ARight, pos=(-0.1, 0.1),
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.5))
        # State setup
        self.objects = objects
        self.sim = simulator
        # Keys
        self.accept("escape", sys.exit)
        self.accept("r", self.reset_sim)
        self.accept("space", self.toggle_sim)
        self.accept("s", self.step_once)
        self.accept("f1", self.toggle_wireframe)
        self.accept("f2", self.toggle_particles)
        self.accept("arrow_up", self.method_selector, [True])
        self.accept("arrow_down", self.method_selector, [False])
        # Reposition camera, with -z as front
        base.useDrive()
        base.useTrackball()
        base.cam.setPos(0, 0, 0)
        base.cam.setHpr(0, -90, 0)
        base.cam.setPos(0, 0, 100)
        # Display info
        self.make_label(0, lambda: 'Elapsed time: ' + str(round(self.sim.elapsed_time, 2)))
        self.make_label(1, lambda: 'Steps: ' + str(self.sim.step_counter))
        self.make_instruction(5, "R: reset simulation")
        self.make_instruction(4, "SPACE: stop/continue")
        self.make_instruction(3, "S: step once")
        self.make_instruction(2, "F1: toggle wireframe")
        self.make_instruction(1, "F2: toggle particles")
        self.make_instruction(0, "ESC: quit")
        # General setup
        self.setBackgroundColor((0,0,0, 1))
        r = self.mesh.getRoot()
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
    
    def make_label(self, pos, cb):
        self.stats.append((OnscreenText(text=str(cb()), parent=base.a2dTopLeft, pos=(0.07, -.09 * pos - 0.1),
            fg=(1, 1, 1, 1), align=TextNode.ALeft, mayChange=True, scale=.09), cb))

    def make_instruction(self, pos, content):
        OnscreenText(text=content, parent=base.a2dTopLeft, pos=(0.07, pos * 0.08 - 1.92),
            fg=(1, 1, 1, 1), align=TextNode.ALeft, scale=.055)

    def set_title(self, text):
        self.title.text = text

    def method_selector(self, right):
        m = self.sim.method
        opt = [*self.sim.implemented.keys()]
        if right: next = 0 if opt.index(m) + 1 >= len(opt) else opt.index(m) + 1
        else: next = opt.index(m) - 1
        self.change_method = opt[next]
        self.set_title(opt[next] + " (pending reset: press 'R')")
    
    def show_stats(self):
         for scrtxt, cb in self.stats:
             scrtxt.text = str(cb())

    def toggle_sim(self):
        self.sim_running = not self.sim_running

    def toggle_wireframe(self):
        self.draw_wireframe = not self.draw_wireframe

    def toggle_particles(self):
        self.draw_particles = not self.draw_particles

    def reset_sim(self):
        self.sim.method = self.change_method
        self.set_title(self.sim.method)
        self.sim.reset()

    def step_once(self):
        self.sim_running = False
        self.sim.step()

    def draw(self, task):
        self.show_stats()
        self.mesh.begin(base.cam, render)
        self.drawTriangles()
        if self.draw_particles:
            self.drawParticles()
        if self.draw_wireframe:
            self.drawSprings()
        self.mesh.end()
        return task.cont

    def updateSimMain(self, task):
        """Main thread: simply step once called"""
        if self.sim_running:
            self.sim.step()
        return task.cont

    def updateSim(self, task):
        """Separate thread: unfinished"""
        total = globalClock.getDt()
        if total >= self.sim.get_dt() and self.sim_running:
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