#!/usr/bin/env python3
"""
Educational Smoke Simulation — Pygame Version
==============================================
Converted from PySide6 to Pygame.
Inspired by Sebastian Lague's "Coding Adventure: Fluid Simulation"

Install:  pip install pygame numpy scipy
Run:      python smoke_pygame.py
"""

import sys, time, math
import numpy as np
import pygame

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR MAPS (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
COLORMAPS = {
    "Fire": [
        (0,.0,0,0),(0.12,60,0,0),(0.25,160,25,0),(0.45,255,90,0),
        (0.65,255,190,40),(0.82,255,245,130),(1,255,255,255)],
    "Grayscale": [(0,0,0,0),(1,255,255,255)],
    "Cool": [
        (0,0,0,20),(0.2,0,30,100),(0.45,0,120,200),
        (0.7,60,195,245),(1,220,248,255)],
    "Neon": [
        (0,0,0,0),(0.18,80,0,180),(0.38,200,0,140),
        (0.58,255,50,50),(0.78,255,170,0),(1,255,255,100)],
    "Thermal": [
        (0,0,0,0),(0.18,0,0,160),(0.36,0,160,170),(0.5,0,190,0),
        (0.68,210,210,0),(0.84,255,110,0),(1,255,255,255)],
    "Sunset": [
        (0,0,0,0),(0.14,50,0,70),(0.28,140,15,70),(0.48,220,60,45),
        (0.68,255,150,55),(0.84,255,225,130),(1,255,255,235)],
}

def apply_colormap(values, cmap_name):
    stops = COLORMAPS.get(cmap_name, COLORMAPS["Fire"])
    pos = [s[0] for s in stops]
    cols = np.array([s[1:] for s in stops], dtype=np.float64)
    out = np.zeros((*values.shape, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(values.ravel(), pos, cols[:, c]).reshape(values.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  FLUID SOLVER (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
class FluidSolver:
    def __init__(self, N=100):
        self.N = N
        self.visc = 0.0001; self.diff = 0.0001; self.temp_diff = 0.0002
        self.dt = 0.1; self.buoyancy = 1.0; self.vort_conf = 5.0
        self.iterations = 18; self.decay = 0.997; self.temp_cool = 0.992
        self.sources = []; self.step_count = 0; self.solve_ms = 0.0
        self.debug = {}; self._alloc()

    def _alloc(self):
        s = (self.N+2, self.N+2)
        self.d = np.zeros(s); self.T = np.zeros(s)
        self.u = np.zeros(s); self.v = np.zeros(s)
        self.obstacle = np.zeros(s, dtype=bool)

    def reset(self):
        self._alloc(); self.sources.clear(); self.debug.clear(); self.step_count = 0

    def resize(self, N):
        old = self.N; self.N = N; self._alloc(); self.debug.clear()
        scale = N / max(old, 1)
        for i, s in enumerate(self.sources):
            self.sources[i] = (int(s[0]*scale), int(s[1]*scale), *s[2:])
        self.step_count = 0

    def _bnd(self, b, x):
        N = self.N
        x[0,1:N+1] = -x[1,1:N+1] if b==1 else x[1,1:N+1]
        x[N+1,1:N+1] = -x[N,1:N+1] if b==1 else x[N,1:N+1]
        x[1:N+1,0] = -x[1:N+1,1] if b==2 else x[1:N+1,1]
        x[1:N+1,N+1] = -x[1:N+1,N] if b==2 else x[1:N+1,N]
        x[0,0]=.5*(x[1,0]+x[0,1]); x[0,N+1]=.5*(x[1,N+1]+x[0,N])
        x[N+1,0]=.5*(x[N,0]+x[N+1,1]); x[N+1,N+1]=.5*(x[N,N+1]+x[N+1,N])
        x[self.obstacle] = 0 if b != 0 else x[self.obstacle]

    def _diffuse(self, b, x, x0, diff):
        a = self.dt*diff*self.N*self.N
        if a < 1e-14: x[:]=x0; return
        c = 1+4*a; N = self.N
        for _ in range(self.iterations):
            x[1:N+1,1:N+1] = (x0[1:N+1,1:N+1]+a*(
                x[0:N,1:N+1]+x[2:N+2,1:N+1]+
                x[1:N+1,0:N]+x[1:N+1,2:N+2]))/c
            self._bnd(b, x)

    def _advect(self, b, d, d0, u, v):
        N = self.N; dt0 = self.dt*N
        ii, jj = np.meshgrid(np.arange(1,N+1), np.arange(1,N+1), indexing='ij')
        x = np.clip(ii-dt0*u[1:N+1,1:N+1], .5, N+.5)
        y = np.clip(jj-dt0*v[1:N+1,1:N+1], .5, N+.5)
        i0 = np.floor(x).astype(int); j0 = np.floor(y).astype(int)
        s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1
        i0c=np.clip(i0,0,N+1); i1c=np.clip(i0+1,0,N+1)
        j0c=np.clip(j0,0,N+1); j1c=np.clip(j0+1,0,N+1)
        result = (s0*(t0*d0[i0c,j0c]+t1*d0[i0c,j1c])+s1*(t0*d0[i1c,j0c]+t1*d0[i1c,j1c]))
        obs = self.obstacle[1:N+1,1:N+1]
        d[1:N+1,1:N+1] = np.where(obs, d0[1:N+1,1:N+1], result)
        self._bnd(b, d)

    def _project(self, u, v):
        N = self.N; h = 1.0/N
        p = np.zeros((N+2,N+2)); div = np.zeros((N+2,N+2))
        div[1:N+1,1:N+1] = -.5*h*(
            u[2:N+2,1:N+1]-u[0:N,1:N+1]+v[1:N+1,2:N+2]-v[1:N+1,0:N])
        div[self.obstacle] = 0; self._bnd(0, div)
        for _ in range(self.iterations):
            p[1:N+1,1:N+1] = (div[1:N+1,1:N+1]+p[0:N,1:N+1]+p[2:N+2,1:N+1]+p[1:N+1,0:N]+p[1:N+1,2:N+2])/4
            p[self.obstacle] = 0; self._bnd(0, p)
        u[1:N+1,1:N+1] -= .5*N*(p[2:N+2,1:N+1]-p[0:N,1:N+1])
        v[1:N+1,1:N+1] -= .5*N*(p[1:N+1,2:N+2]-p[1:N+1,0:N])
        self._bnd(1,u); self._bnd(2,v)
        return p

    def _apply_forces(self):
        N = self.N
        if self.buoyancy > 0: self.v[1:N+1,1:N+1] -= self.buoyancy*self.T[1:N+1,1:N+1]*self.dt
        if self.vort_conf > 0: self._vorticity_confinement(self.vort_conf)

    def _vorticity_confinement(self, eps):
        N = self.N; h = 1.0/N
        w = .5*((self.v[2:N+2,1:N+1]-self.v[0:N,1:N+1])-(self.u[1:N+1,2:N+2]-self.u[1:N+1,0:N]))
        aw = np.abs(w)
        dwdx = .5*(aw[2:,1:-1]-aw[:-2,1:-1]); dwdy = .5*(aw[1:-1,2:]-aw[1:-1,:-2])
        length = np.sqrt(dwdx*dwdx+dwdy*dwdy)+1e-10; dwdx/=length; dwdy/=length
        w_sub = w[1:-1,1:-1]
        self.u[2:N,2:N] += eps*h*dwdy*w_sub*self.dt
        self.v[2:N,2:N] -= eps*h*dwdx*w_sub*self.dt

    def _apply_obstacles(self):
        self.u[self.obstacle] = 0; self.v[self.obstacle] = 0

    def _inject(self, field, x, y, amt, r=3):
        N = self.N
        i0, i1 = max(1, x-r), min(N, x+r); j0, j1 = max(1, y-r), min(N, y+r)
        if i0 > i1 or j0 > j1: return
        ii, jj = np.meshgrid(np.arange(i0,i1+1), np.arange(j0,j1+1), indexing='ij')
        dd = np.sqrt(((ii-x)**2+(jj-y)**2).astype(np.float64))
        field[i0:i1+1, j0:j1+1] += np.where(dd<=r, amt*(1-dd/(r+1)), 0)

    def add_density(self, x, y, amt, r=3): self._inject(self.d, x, y, amt, r)
    def add_velocity(self, x, y, vx, vy, r=3): self._inject(self.u, x, y, vx, r); self._inject(self.v, x, y, vy, r)
    def add_temperature(self, x, y, amt, r=3): self._inject(self.T, x, y, amt, r)

    def set_obstacle(self, x, y, r=2, value=True):
        N = self.N; i0, i1 = max(1, x-r), min(N, x+r); j0, j1 = max(1, y-r), min(N, y+r)
        if i0 > i1 or j0 > j1: return
        ii, jj = np.meshgrid(np.arange(i0,i1+1), np.arange(j0,j1+1), indexing='ij')
        dd = np.sqrt(((ii-x)**2+(jj-y)**2).astype(np.float64))
        self.obstacle[i0:i1+1, j0:j1+1] = np.where(dd<=r, value, self.obstacle[i0:i1+1, j0:j1+1])

    def _sanitize(self):
        for f in (self.d, self.T, self.u, self.v):
            bad = ~np.isfinite(f)
            if bad.any(): f[bad] = 0

    def step(self):
        t0 = time.perf_counter(); N = self.N
        for sx,sy,sr,svx,svy,sT in self.sources:
            ix, iy = int(np.clip(sx,1,N)), int(np.clip(sy,1,N))
            self.add_density(ix, iy, sr, r=2); self.add_velocity(ix, iy, svx, svy, r=2)
            if sT != 0: self.add_temperature(ix, iy, sT, r=2)
        self._apply_forces(); self._apply_obstacles()
        u0=self.u.copy(); v0=self.v.copy()
        self._diffuse(1,self.u,u0,self.visc); self._diffuse(2,self.v,v0,self.visc)
        self._apply_obstacles(); pressure = self._project(self.u, self.v)
        u0[:]=self.u; v0[:]=self.v
        self._advect(1,self.u,u0,u0,v0); self._advect(2,self.v,v0,u0,v0)
        self._apply_obstacles(); self._project(self.u, self.v)
        d0=self.d.copy(); self._diffuse(0,self.d,d0,self.diff); d0[:]=self.d
        self._advect(0,self.d,d0,self.u,self.v)
        T0=self.T.copy(); self._diffuse(0,self.T,T0,self.temp_diff); T0[:]=self.T
        self._advect(0,self.T,T0,self.u,self.v)
        self.T *= self.temp_cool; self.T = np.clip(self.T, 0, None)
        self.d *= self.decay; self.d = np.clip(self.d, 0, None)
        self.u = np.clip(self.u,-120,120); self.v = np.clip(self.v,-120,120)
        self._sanitize()
        self.debug['pressure'] = pressure; self.debug['vorticity'] = self.vorticity_field()
        self.step_count += 1; self.solve_ms = (time.perf_counter()-t0)*1000

    def vorticity_field(self):
        N = self.N; w = np.zeros((N+2,N+2))
        w[1:N+1,1:N+1] = .5*((self.v[2:N+2,1:N+1]-self.v[0:N,1:N+1])-(self.u[1:N+1,2:N+2]-self.u[1:N+1,0:N]))
        return w

    def max_velocity(self): return np.sqrt(self.u**2+self.v**2).max()
    def total_density(self): return self.d[1:self.N+1,1:self.N+1].sum()


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR CONSTANTS  (Catppuccin Mocha palette)
# ══════════════════════════════════════════════════════════════════════════════
C_BASE      = (30, 30, 46)
C_MANTLE    = (24, 24, 37)
C_SURF0     = (49, 50, 68)
C_SURF1     = (69, 71, 90)
C_SURF2     = (88, 91, 112)
C_TEXT      = (205, 214, 244)
C_SUBTEXT   = (166, 173, 200)
C_BLUE      = (137, 180, 250)
C_GREEN     = (166, 227, 161)
C_PEACH     = (250, 179, 135)
C_YELLOW    = (249, 226, 175)
C_RED       = (243, 139, 168)
C_CANVAS_BG = (14, 14, 24)
C_TOOLBAR   = (24, 24, 37)

# ══════════════════════════════════════════════════════════════════════════════
#  FONTS HELPER
# ══════════════════════════════════════════════════════════════════════════════
FONT_NAME = None
MONO_NAME = None

def get_font(size, bold=False):
    global FONT_NAME
    if FONT_NAME is None:
        for name in ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", None]:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                FONT_NAME = name
                return f
            except:
                continue
        FONT_NAME = None
    return pygame.font.SysFont(FONT_NAME, size, bold=bold)

def get_mono_font(size):
    global MONO_NAME
    if MONO_NAME is None:
        for name in ["Consolas", "DejaVu Sans Mono", "Courier New", None]:
            try:
                f = pygame.font.SysFont(name, size)
                MONO_NAME = name
                return f
            except:
                continue
        MONO_NAME = None
    return pygame.font.SysFont(MONO_NAME, size)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE PYGAME UI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, x, y, w, h, text, toggle=False, color=C_SURF0, active_color=C_BLUE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.toggle = toggle
        self.color = color
        self.active_color = active_color
        self.active = False
        self.hovered = False
        self.clicked = False

    def handle_event(self, ev):
        self.clicked = False
        if ev.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.clicked = True
                if self.toggle:
                    self.active = not self.active
                return True
        return False

    def draw(self, surf):
        if self.toggle and self.active:
            col = self.active_color
            tcol = C_BASE
        elif self.hovered:
            col = C_SURF1
            tcol = C_TEXT
        else:
            col = self.color
            tcol = C_TEXT
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        pygame.draw.rect(surf, C_SURF1, self.rect, 1, border_radius=5)
        font = get_font(13, bold=(self.toggle and self.active))
        txt = font.render(self.text, True, tcol)
        surf.blit(txt, (self.rect.centerx - txt.get_width()//2,
                        self.rect.centery - txt.get_height()//2))


class Slider:
    def __init__(self, x, y, w, label, lo, hi, value, highlight=False):
        self.x = x; self.y = y; self.w = w; self.h = 36
        self.label = label; self.lo = lo; self.hi = hi
        self.value = value; self.highlight = highlight
        self.dragging = False
        self._bar_rect = pygame.Rect(x + 100, y + 16, w - 140, 6)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            hit = self._bar_rect.inflate(10, 14)
            if hit.collidepoint(ev.pos):
                self.dragging = True
                self._update_from_mouse(ev.pos[0])
                return True
        elif ev.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(ev.pos[0])
            return True
        return False

    def _update_from_mouse(self, mx):
        t = (mx - self._bar_rect.x) / max(self._bar_rect.w, 1)
        t = max(0.0, min(1.0, t))
        self.value = self.lo + t * (self.hi - self.lo)

    @property
    def int_value(self):
        return int(round(self.value))

    def draw(self, surf):
        label_col = C_YELLOW if self.highlight else C_TEXT
        font = get_font(12, bold=self.highlight)
        lbl = font.render(self.label, True, label_col)
        surf.blit(lbl, (self.x, self.y + 2))
        vfont = get_mono_font(11)
        vtxt = vfont.render(str(self.int_value), True, label_col)
        surf.blit(vtxt, (self.x + self.w - 30, self.y + 2))
        pygame.draw.rect(surf, C_SURF1, self._bar_rect, border_radius=3)
        t = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        fill_r = pygame.Rect(self._bar_rect.x, self._bar_rect.y,
                             int(self._bar_rect.w * t), self._bar_rect.h)
        pygame.draw.rect(surf, C_BLUE, fill_r, border_radius=3)
        hx = self._bar_rect.x + int(self._bar_rect.w * t)
        hy = self._bar_rect.centery
        pygame.draw.circle(surf, C_BLUE, (hx, hy), 7)
        if self.highlight:
            pygame.draw.circle(surf, C_YELLOW, (hx, hy), 7, 2)


class Checkbox:
    def __init__(self, x, y, label, checked=False):
        self.rect = pygame.Rect(x, y, 18, 18)
        self.label = label; self.checked = checked

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            click_area = pygame.Rect(self.rect.x, self.rect.y, 200, 20)
            if click_area.collidepoint(ev.pos):
                self.checked = not self.checked
                return True
        return False

    def draw(self, surf):
        pygame.draw.rect(surf, C_SURF0, self.rect, border_radius=3)
        pygame.draw.rect(surf, C_SURF2, self.rect, 2, border_radius=3)
        if self.checked:
            inner = self.rect.inflate(-6, -6)
            pygame.draw.rect(surf, C_BLUE, inner, border_radius=2)
        font = get_font(12)
        lbl = font.render(self.label, True, C_TEXT)
        surf.blit(lbl, (self.rect.right + 6, self.rect.y + 1))


class TabBar:
    def __init__(self, x, y, w, tabs):
        self.x = x; self.y = y; self.w = w
        self.tabs = tabs; self.selected = 0
        self.tab_w = w // len(tabs)
        self._rects = [pygame.Rect(x + i * self.tab_w, y, self.tab_w, 30) for i in range(len(tabs))]

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self._rects):
                if r.collidepoint(ev.pos):
                    self.selected = i; return True
        return False

    def draw(self, surf):
        for i, (r, name) in enumerate(zip(self._rects, self.tabs)):
            if i == self.selected:
                col = C_BASE; border_col = C_BLUE
            else:
                col = C_MANTLE; border_col = C_SURF0
            pygame.draw.rect(surf, col, r)
            pygame.draw.rect(surf, border_col, r, 1)
            if i == self.selected:
                pygame.draw.line(surf, C_BLUE, (r.x + 1, r.bottom - 2), (r.right - 1, r.bottom - 2), 2)
            font = get_font(12, bold=(i == self.selected))
            tc = C_TEXT if i == self.selected else C_SUBTEXT
            txt = font.render(name, True, tc)
            surf.blit(txt, (r.centerx - txt.get_width()//2, r.centery - txt.get_height()//2))


# ══════════════════════════════════════════════════════════════════════════════
#  LESSONS / PRESETS DATA
# ══════════════════════════════════════════════════════════════════════════════
LESSONS = [
    { "title": "1. What is Fluid Sim?", "preset": "candle", "highlights": ["buoyancy"],
      "text": ["What is Fluid Simulation?",
               "",
               "Solving the Navier-Stokes equations",
               "  governing weather, oceans, and smoke.",
               "",
               "Why is it hard?",
               "  Non-linear + incompressibility constraint.",
               "",
               "Stam's Insight: Trade accuracy for",
               "  guaranteed stability.",
               "",
               ">> Click 'Run This Demo' for a rising plume.",
               ">> Drag mouse on canvas to interact!"] },
    { "title": "2. Navier-Stokes Eqns", "preset": "default", "highlights": ["viscosity"],
      "text": ["Navier-Stokes Equations",
               "",
               "  du/dt = -(u.grad)u + v*Lap(u) - grad(p) + f",
               "",
               "  -(u.grad)u    Advection",
               "  v*Lap(u)      Viscous Diffusion",
               "  -grad(p)      Pressure",
               "  f             External Forces",
               "",
               "  div(u) = 0    (Incompressibility)"] },
    { "title": "3. Advection", "preset": "advection", "highlights": ["dt"],
      "text": ["Advection",
               "",
               "Carrying quantities along the velocity field.",
               "",
               "Naive:  x_new = x + dt*u(x)",
               "  >> Unstable!",
               "",
               "Semi-Lagrangian:",
               "  Trace backwards. Sample via interpolation.",
               "  >> Stable!"] },
    { "title": "4. Diffusion", "preset": "diffusion", "highlights": ["viscosity", "diffusion"],
      "text": ["Diffusion",
               "",
               "Velocity spreads via viscosity (v)",
               "Density spreads via diffusion (k)",
               "",
               "Implicit Solve: Gauss-Seidel relaxation.",
               "  >> Unconditionally stable."] },
    { "title": "5. Pressure", "preset": "pressure", "highlights": [],
      "text": ["Pressure",
               "",
               "Fixing divergence via Helmholtz-Hodge",
               "  decomposition.",
               "",
               "  1. Compute div(u)",
               "  2. Solve Poisson: Lap(p) = div(u)",
               "  3. Subtract grad(p)",
               "",
               "Enable Pressure Overlay (P key) to see!"] },
    { "title": "6. Forces", "preset": "vorticity", "highlights": ["buoyancy", "vorticity"],
      "text": ["Forces",
               "",
               "Buoyancy: Temperature drives upward force.",
               "Vorticity Confinement: Re-injects lost",
               "  swirl energy.",
               "",
               "Try setting vorticity to 0 vs 20!"] },
    { "title": "7. Obstacles", "preset": "obstacle", "highlights": [],
      "text": ["Obstacles",
               "",
               "Walls enforce zero velocity and alter",
               "  pressure.",
               "",
               "Toggle Wall Mode in toolbar:",
               "  Left-draw wall, Right-erase.",
               "",
               "Press W key to toggle Wall Mode."] },
    { "title": "8. Playground", "preset": "playground", "highlights": [],
      "text": ["Playground",
               "",
               "Experiment freely!",
               "",
               "Left-drag:  Smoke + Heat + Push",
               "Right-drag: Wind only",
               "",
               "Keyboard shortcuts:",
               "  Space  Pause/Play",
               "  R      Reset",
               "  S      Single step",
               "  V/W/P/T  Toggle overlays"] },
]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class SmokeApp:
    PANEL_W = 320
    TOOLBAR_H = 42
    STATUS_H = 24

    def __init__(self):
        pygame.init()
        self.W, self.H = 1200, 780
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("Educational Smoke Simulation - Fluid Dynamics (Pygame)")
        self.clock = pygame.time.Clock()

        self.solver = FluidSolver(N=100)
        self.paused = False
        self.cmap_name = "Fire"
        self.show_velocity = False
        self.show_vorticity = False
        self.show_pressure = False
        self.show_temperature = False
        self.show_colorbar = True
        self.source_mode = False
        self.obstacle_mode = False
        self.steps_per_frame = 1
        self.current_lesson = 0

        self.mouse_down = None
        self.last_mouse = None
        self._frame_times = []
        self._cmap_rects = []
        self._grid_rects = []
        self._steps_rects = []

        self._build_ui()
        self._apply_preset("candle")

    def _build_ui(self):
        bx = 8
        self.btn_pause = Button(bx, 6, 80, 30, "Pause", toggle=True); bx += 88
        self.btn_step  = Button(bx, 6, 70, 30, "Step"); bx += 78
        self.btn_reset = Button(bx, 6, 80, 30, "Reset"); bx += 92
        self.btn_source = Button(bx, 6, 80, 30, "Source", toggle=True); bx += 88
        self.btn_wall   = Button(bx, 6, 70, 30, "Wall", toggle=True)
        self.toolbar_buttons = [self.btn_pause, self.btn_step, self.btn_reset,
                                self.btn_source, self.btn_wall]

        panel_x = self.W - self.PANEL_W
        self.tab_bar = TabBar(panel_x, self.TOOLBAR_H, self.PANEL_W,
                              ["Learn", "Physics", "Visuals"])

        sx = panel_x + 12
        self.sliders = {
            "viscosity":  Slider(sx, 0, 290, "Viscosity (v)",    0, 50,  1),
            "diffusion":  Slider(sx, 0, 290, "Diffusion (k)",    0, 50,  1),
            "buoyancy":   Slider(sx, 0, 290, "Buoyancy (a)",     0, 100, 10),
            "vorticity":  Slider(sx, 0, 290, "Vorticity (e)",    0, 250, 50),
            "dt":         Slider(sx, 0, 290, "Time Step",        1, 30,  10),
            "decay":      Slider(sx, 0, 290, "Decay (permille)",980,1000,997),
        }

        vx = panel_x + 16
        self.cb_velocity    = Checkbox(vx, 0, "Velocity", False)
        self.cb_vorticity   = Checkbox(vx + 105, 0, "Vorticity", False)
        self.cb_pressure    = Checkbox(vx, 0, "Pressure", False)
        self.cb_temperature = Checkbox(vx + 105, 0, "Temp", False)
        self.cb_colorbar    = Checkbox(vx, 0, "Colour bar", True)

        self.cmap_names = list(COLORMAPS.keys())
        self.cmap_index = 0
        self.grid_options = [64, 100, 128, 160]
        self.grid_index = 1
        self.steps_options = [1, 2, 3, 4, 5, 6, 7, 8]
        self.steps_index = 0

        self.btn_prev = Button(panel_x + 12, 0, 32, 30, "<")
        self.btn_next = Button(panel_x + self.PANEL_W - 44, 0, 32, 30, ">")
        self.btn_run_demo = Button(panel_x + 12, 0, self.PANEL_W - 24, 34,
                                    "Run This Demo", color=(30, 102, 64),
                                    active_color=(42, 128, 80))

    def _canvas_rect(self):
        cw = self.W - self.PANEL_W - 30
        ch = self.H - self.TOOLBAR_H - self.STATUS_H - 20
        s = min(cw, ch)
        x = 10 + (cw - s) // 2
        y = self.TOOLBAR_H + 10 + (ch - s) // 2
        return pygame.Rect(x, y, s, s)

    def _screen_to_grid(self, pos):
        r = self._canvas_rect()
        N = self.solver.N
        gi = int(np.clip((pos[0] - r.x) / r.w * (N + 2), 1, N))
        gj = int(np.clip((pos[1] - r.y) / r.h * (N + 2), 1, N))
        return gi, gj

    # ── Presets ─────────────────────────────────────────────────────────
    def _apply_preset(self, name):
        self.solver.reset()
        N = self.solver.N; cx = N // 2
        presets = {
            "candle":     dict(buoyancy=2.0,vort_conf=3.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-6,12)],cmap="Fire"),
            "default":    dict(buoyancy=1.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[],cmap="Fire"),
            "advection":  dict(buoyancy=1.0,vort_conf=5.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,sources=[(cx,N-5,20,0,-5,10)],cmap="Cool"),
            "diffusion":  dict(buoyancy=1.5,vort_conf=0.0,visc=0.005,diff=0.005,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-5,12)],cmap="Grayscale"),
            "pressure":   dict(buoyancy=2.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-6,12)],cmap="Fire",overlays={"pressure":True}),
            "vorticity":  dict(buoyancy=2.0,vort_conf=20.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,sources=[(cx,N-5,25,0,-5,12)],cmap="Neon",overlays={"vorticity":True}),
            "obstacle":   dict(buoyancy=2.0,vort_conf=8.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,30,0,-7,15)],cmap="Thermal",obstacles="wall_with_gap"),
            "playground": dict(buoyancy=1.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[],cmap="Fire"),
        }
        p = presets.get(name, presets["default"])
        self.solver.buoyancy = p["buoyancy"]
        self.solver.vort_conf = p["vort_conf"]
        self.solver.visc = p["visc"]
        self.solver.diff = p["diff"]
        self.solver.decay = p["decay"]
        self.solver.dt = p["dt"]
        for s in p.get("sources", []):
            self.solver.sources.append(s)
        if p.get("obstacles") == "wall_with_gap":
            for i in range(1, N + 1):
                for j in range(N // 2 - 2, N // 2 + 2):
                    if abs(i - cx) > 6:
                        self.solver.obstacle[i, j] = True
        self.show_pressure = p.get("overlays", {}).get("pressure", False)
        self.show_vorticity = p.get("overlays", {}).get("vorticity", False)
        self.cb_pressure.checked = self.show_pressure
        self.cb_vorticity.checked = self.show_vorticity
        cname = p.get("cmap", "Fire")
        self.cmap_name = cname
        if cname in self.cmap_names:
            self.cmap_index = self.cmap_names.index(cname)
        self._sync_sliders_from_solver()

    def _sync_sliders_from_solver(self):
        s = self.solver
        self.sliders["viscosity"].value = int(s.visc / 1e-4)
        self.sliders["diffusion"].value = int(s.diff / 1e-4)
        self.sliders["buoyancy"].value = int(s.buoyancy * 10)
        self.sliders["vorticity"].value = int(s.vort_conf * 10)
        self.sliders["dt"].value = int(s.dt * 100)
        self.sliders["decay"].value = int(s.decay * 1000)

    def _sync_solver_from_sliders(self):
        s = self.solver
        s.visc     = self.sliders["viscosity"].value * 1e-4
        s.diff     = self.sliders["diffusion"].value * 1e-4
        s.buoyancy = self.sliders["buoyancy"].value / 10.0
        s.vort_conf= self.sliders["vorticity"].value / 10.0
        s.dt       = self.sliders["dt"].value / 100.0
        s.decay    = self.sliders["decay"].value / 1000.0

    def _highlight_sliders(self, keys):
        key_map = {"buoyancy": "buoyancy", "viscosity": "viscosity",
                   "vorticity": "vorticity", "diffusion": "diffusion"}
        active = set(key_map.get(k, k) for k in keys)
        for k, sl in self.sliders.items():
            sl.highlight = k in active

    # ── Rendering ───────────────────────────────────────────────────────
    def _render_canvas(self, surf):
        rect = self._canvas_rect()
        N = self.solver.N

        pygame.draw.rect(surf, (8, 8, 14), rect, border_radius=4)
        pygame.draw.rect(surf, C_SURF1, rect.inflate(4, 4), 2, border_radius=6)

        d = self.solver.d[1:N+1, 1:N+1].T
        dmax = max(d.max(), 1e-6)
        dn = np.clip(d / dmax, 0, 1) ** 0.6
        rgb = np.ascontiguousarray(apply_colormap(dn, self.cmap_name))

        rgb_transposed = np.transpose(rgb, (1, 0, 2))
        grid_surf = pygame.surfarray.make_surface(rgb_transposed)
        grid_surf = pygame.transform.smoothscale(grid_surf, (rect.w, rect.h))
        surf.blit(grid_surf, rect.topleft)

        if self.show_temperature:
            self._draw_overlay(surf, rect, N, self._temperature_rgb, 0.35)
        if self.show_pressure:
            self._draw_overlay(surf, rect, N, self._pressure_rgb, 0.30)
        if self.show_vorticity:
            self._draw_overlay(surf, rect, N, self._vorticity_rgb, 0.35)
        self._draw_obstacles(surf, rect, N)
        if self.show_velocity:
            self._draw_velocity_arrows(surf, rect, N)

        for sx, sy, *_ in self.solver.sources:
            mx = rect.x + sx / (N + 2) * rect.w
            my = rect.y + sy / (N + 2) * rect.h
            pygame.draw.circle(surf, (255, 220, 50), (int(mx), int(my)), 7, 2)
            pygame.draw.line(surf, (255, 220, 50), (int(mx)-4, int(my)), (int(mx)+4, int(my)), 1)
            pygame.draw.line(surf, (255, 220, 50), (int(mx), int(my)-4), (int(mx), int(my)+4), 1)

        if self.source_mode:
            txt = get_font(13, bold=True).render("Source Mode", True, (255, 220, 50))
            surf.blit(txt, (rect.centerx - txt.get_width()//2, rect.y + 8))
        elif self.obstacle_mode:
            txt = get_font(13, bold=True).render("Wall Mode (R-click: Erase)", True, C_RED)
            surf.blit(txt, (rect.centerx - txt.get_width()//2, rect.y + 8))

        if self.show_colorbar:
            self._draw_colorbar(surf, rect, dmax)

    def _draw_overlay(self, surf, rect, N, rgb_fn, alpha):
        rgb = rgb_fn(N)
        rgb_t = np.transpose(rgb, (1, 0, 2))
        overlay = pygame.surfarray.make_surface(rgb_t)
        overlay = pygame.transform.smoothscale(overlay, (rect.w, rect.h))
        overlay.set_alpha(int(alpha * 255))
        surf.blit(overlay, rect.topleft)

    def _temperature_rgb(self, N):
        T = self.solver.T[1:N+1, 1:N+1].T
        tm = max(T.max(), 1e-6); tn = np.clip(T / tm, 0, 1)
        r = (tn * 220 + 20).astype(np.uint8)
        g = (tn * 120 + 10).astype(np.uint8)
        b = np.full_like(r, 15, dtype=np.uint8)
        return np.stack([r, g, b], axis=-1)

    def _pressure_rgb(self, N):
        pf = self.solver.debug.get('pressure')
        if pf is None:
            return np.zeros((N, N, 3), dtype=np.uint8)
        d = pf[1:N+1, 1:N+1].T
        mx = max(np.abs(d).max(), 1e-8); dn = d / mx
        pos = np.clip(dn, 0, 1); neg = np.clip(-dn, 0, 1)
        r = (pos * 200 + 30).astype(np.uint8)
        g = np.full_like(r, 30, dtype=np.uint8)
        b = (neg * 200 + 30).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    def _vorticity_rgb(self, N):
        wf = self.solver.debug.get('vorticity')
        if wf is None:
            return np.zeros((N, N, 3), dtype=np.uint8)
        w = wf[1:N+1, 1:N+1].T
        wn = np.clip(w / max(np.abs(w).max(), 1e-8), -1, 1)
        pos = np.clip(wn, 0, 1); neg = np.clip(-wn, 0, 1)
        r = (pos * 220 + 20).astype(np.uint8)
        g = np.full_like(r, 20, dtype=np.uint8)
        b = (neg * 220 + 20).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    def _draw_obstacles(self, surf, rect, N):
        obs = self.solver.obstacle[1:N+1, 1:N+1].T
        if not obs.any():
            return
        rgba = np.zeros((N, N, 4), dtype=np.uint8)
        rgba[obs, 0] = 55; rgba[obs, 1] = 60; rgba[obs, 2] = 75; rgba[obs, 3] = 220
        try:
            from scipy.ndimage import binary_dilation
            edge = binary_dilation(obs, iterations=1) & ~obs
            if edge.any():
                rgba[edge, 0] = 100; rgba[edge, 1] = 110; rgba[edge, 2] = 140; rgba[edge, 3] = 180
        except ImportError:
            pass
        rgb_t = np.transpose(rgba[:, :, :3], (1, 0, 2))
        alpha_t = rgba[:, :, 3].T
        pix_array = np.zeros((N, N, 4), dtype=np.uint8)
        pix_array[:, :, :3] = rgb_t
        pix_array[:, :, 3] = alpha_t
        obs_surf = pygame.image.frombuffer(
            pix_array.tobytes(), (N, N), "RGBA").convert_alpha()
        obs_surf = pygame.transform.smoothscale(obs_surf, (rect.w, rect.h))
        surf.blit(obs_surf, rect.topleft)

    def _draw_velocity_arrows(self, surf, rect, N):
        step = max(N // 22, 1)
        vmax = max(self.solver.max_velocity(), 0.1)
        sc = min(rect.w, rect.h) / (N + 2) * 0.8 / vmax * step
        sc = max(0.5, min(sc, 30))
        for i in range(1, N + 1, step):
            for j in range(1, N + 1, step):
                if self.solver.obstacle[i, j]:
                    continue
                ux = self.solver.u[i, j]; vy = self.solver.v[i, j]
                if abs(ux) + abs(vy) < vmax * 0.01:
                    continue
                sx = rect.x + i / (N + 2) * rect.w
                sy = rect.y + j / (N + 2) * rect.h
                ex = sx + ux * sc; ey = sy + vy * sc
                pygame.draw.line(surf, (130, 200, 255),
                                 (int(sx), int(sy)), (int(ex), int(ey)), 1)

    def _draw_colorbar(self, surf, rect, dmax):
        cw, ch = 18, 120
        x0 = rect.right - cw - 15
        y0 = rect.bottom - ch - 30
        bg = pygame.Rect(x0 - 6, y0 - 18, cw + 52, ch + 30)
        bg_surf = pygame.Surface((bg.w, bg.h), pygame.SRCALPHA)
        bg_surf.fill((20, 20, 35, 180))
        surf.blit(bg_surf, bg.topleft)
        pygame.draw.rect(surf, C_SURF0, bg, border_radius=4)
        lbl = get_mono_font(11).render("rho", True, C_TEXT)
        surf.blit(lbl, (x0 - 2, y0 - 5))
        for i in range(ch):
            t = 1.0 - i / ch
            rgb = apply_colormap(np.array([[t]]), self.cmap_name)[0, 0]
            pygame.draw.line(surf, (int(rgb[0]), int(rgb[1]), int(rgb[2])),
                             (x0, y0 + i), (x0 + cw, y0 + i))
        pygame.draw.rect(surf, C_TEXT, (x0 - 1, y0 - 1, cw + 2, ch + 2), 1)
        vfont = get_mono_font(10)
        top_lbl = vfont.render(f"{dmax:.0f}", True, C_TEXT)
        bot_lbl = vfont.render("0", True, C_TEXT)
        surf.blit(top_lbl, (x0 + cw + 4, y0 + 3))
        surf.blit(bot_lbl, (x0 + cw + 4, y0 + ch - 8))

    def _render_toolbar(self, surf):
        pygame.draw.rect(surf, C_TOOLBAR, (0, 0, self.W, self.TOOLBAR_H))
        pygame.draw.line(surf, C_SURF0, (0, self.TOOLBAR_H - 1), (self.W, self.TOOLBAR_H - 1))
        for btn in self.toolbar_buttons:
            btn.draw(surf)
        lesson = LESSONS[self.current_lesson]
        lbl = get_font(12, bold=True).render(lesson["title"], True, C_GREEN)
        surf.blit(lbl, (520, 14))

    def _render_statusbar(self, surf):
        y = self.H - self.STATUS_H
        pygame.draw.rect(surf, C_TOOLBAR, (0, y, self.W, self.STATUS_H))
        pygame.draw.line(surf, C_SURF0, (0, y), (self.W, y))
        fps = len(self._frame_times)
        mv = self.solver.max_velocity()
        td = self.solver.total_density()
        sm = self.solver.solve_ms
        sc = self.solver.step_count
        ns = len(self.solver.sources)
        status = (f"FPS: {fps}  |  Solve: {sm:.1f}ms  |  "
                  f"Grid: {self.solver.N}^2  |  Step: {sc}  |  "
                  f"Max|v|: {mv:.1f}  |  Sum(rho): {td:.0f}  |  "
                  f"Sources: {ns}")
        txt = get_mono_font(11).render(status, True, C_SUBTEXT)
        surf.blit(txt, (10, y + 5))

    def _render_panel(self, surf):
        px = self.W - self.PANEL_W
        panel_rect = pygame.Rect(px, self.TOOLBAR_H, self.PANEL_W,
                                 self.H - self.TOOLBAR_H - self.STATUS_H)
        pygame.draw.rect(surf, C_BASE, panel_rect)
        pygame.draw.line(surf, C_SURF0, (px, self.TOOLBAR_H),
                         (px, self.H - self.STATUS_H), 2)
        self.tab_bar.draw(surf)
        tab_y = self.TOOLBAR_H + 34
        if self.tab_bar.selected == 0:
            self._render_learn_tab(surf, px, tab_y)
        elif self.tab_bar.selected == 1:
            self._render_physics_tab(surf, px, tab_y)
        elif self.tab_bar.selected == 2:
            self._render_visuals_tab(surf, px, tab_y)

    def _render_learn_tab(self, surf, px, y):
        self.btn_prev.rect.y = y
        self.btn_next.rect.y = y
        self.btn_prev.draw(surf)
        self.btn_next.draw(surf)
        title = LESSONS[self.current_lesson]["title"]
        tfont = get_font(12, bold=True)
        ttxt = tfont.render(title, True, C_BLUE)
        surf.blit(ttxt, (px + 50, y + 7))

        text_area = pygame.Rect(px + 8, y + 38, self.PANEL_W - 16,
                                self.H - self.STATUS_H - y - 80)
        pygame.draw.rect(surf, C_MANTLE, text_area, border_radius=6)
        pygame.draw.rect(surf, C_SURF0, text_area, 1, border_radius=6)

        lesson = LESSONS[self.current_lesson]
        font = get_font(13)
        bold_font = get_font(14, bold=True)
        ly = text_area.y + 10
        first = True
        for line in lesson["text"]:
            if first:
                txt = bold_font.render(line, True, C_PEACH)
                first = False
            elif line.startswith(">>"):
                txt = font.render(line, True, C_GREEN)
            elif line.startswith("  ") and not line.startswith("   "):
                txt = font.render(line, True, C_BLUE)
            else:
                txt = font.render(line, True, C_TEXT)
            surf.blit(txt, (text_area.x + 12, ly))
            ly += 20

        self.btn_run_demo.rect.y = text_area.bottom + 8
        self.btn_run_demo.rect.x = px + 12
        self.btn_run_demo.rect.w = self.PANEL_W - 24
        self.btn_run_demo.draw(surf)

    def _render_physics_tab(self, surf, px, y):
        font_title = get_font(13, bold=True)
        title = font_title.render("Parameters", True, C_BLUE)
        surf.blit(title, (px + 14, y + 4))
        sy = y + 30
        for key in ["viscosity", "diffusion", "buoyancy", "vorticity", "dt", "decay"]:
            sl = self.sliders[key]
            sl.x = px + 12
            sl.y = sy
            sl.w = self.PANEL_W - 24
            sl._bar_rect = pygame.Rect(sl.x + 100, sl.y + 16, sl.w - 140, 6)
            sl.draw(surf)
            sy += 48

    def _render_visuals_tab(self, surf, px, y):
        font_title = get_font(13, bold=True)
        font = get_font(12)
        small_font = get_font(11)

        cy = y + 8
        lbl = font.render("Colour Map:", True, C_TEXT)
        surf.blit(lbl, (px + 14, cy))
        cy += 24
        self._cmap_rects = []
        bw = (self.PANEL_W - 40) // 3
        for i, name in enumerate(self.cmap_names):
            row, col = divmod(i, 3)
            r = pygame.Rect(px + 12 + col * (bw + 6), cy + row * 32, bw, 26)
            self._cmap_rects.append(r)
            if i == self.cmap_index:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=5)
                tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=5)
                pygame.draw.rect(surf, C_SURF1, r, 1, border_radius=5)
                tc = C_TEXT
            txt = small_font.render(name, True, tc)
            surf.blit(txt, (r.centerx - txt.get_width()//2, r.centery - txt.get_height()//2))

        oy = cy + 80
        lbl = font_title.render("Overlays", True, C_BLUE)
        surf.blit(lbl, (px + 14, oy)); oy += 26
        self.cb_velocity.rect.topleft = (px + 16, oy)
        self.cb_vorticity.rect.topleft = (px + 16 + 130, oy)
        self.cb_velocity.draw(surf); self.cb_vorticity.draw(surf); oy += 26
        self.cb_pressure.rect.topleft = (px + 16, oy)
        self.cb_temperature.rect.topleft = (px + 16 + 130, oy)
        self.cb_pressure.draw(surf); self.cb_temperature.draw(surf); oy += 26
        self.cb_colorbar.rect.topleft = (px + 16, oy)
        self.cb_colorbar.draw(surf); oy += 36

        lbl = font_title.render("Grid Size:", True, C_BLUE)
        surf.blit(lbl, (px + 14, oy)); oy += 24
        self._grid_rects = []
        gw = 55
        for i, g in enumerate(self.grid_options):
            r = pygame.Rect(px + 16 + i * (gw + 6), oy, gw, 26)
            self._grid_rects.append(r)
            if i == self.grid_index:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=4)
                tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=4)
                tc = C_TEXT
            txt = small_font.render(str(g), True, tc)
            surf.blit(txt, (r.centerx - txt.get_width()//2, r.centery - txt.get_height()//2))
        oy += 40

        lbl = font_title.render("Steps / Frame:", True, C_BLUE)
        surf.blit(lbl, (px + 14, oy)); oy += 24
        self._steps_rects = []
        sw = 30
        for i, s in enumerate(self.steps_options):
            r = pygame.Rect(px + 16 + i * (sw + 4), oy, sw, 24)
            self._steps_rects.append(r)
            if i == self.steps_index:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=4)
                tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=4)
                tc = C_TEXT
            txt = small_font.render(str(s), True, tc)
            surf.blit(txt, (r.centerx - txt.get_width()//2, r.centery - txt.get_height()//2))

    # ── Event Handling ──────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)

            for btn in self.toolbar_buttons:
                btn.handle_event(ev)

            if self.btn_pause.clicked:
                self.paused = self.btn_pause.active
                self.btn_pause.text = "Play" if self.paused else "Pause"
            if self.btn_step.clicked:
                self.solver.step()
            if self.btn_reset.clicked:
                self.solver.reset()
            if self.btn_source.clicked:
                self.source_mode = self.btn_source.active
                self.obstacle_mode = False
                self.btn_wall.active = False
            if self.btn_wall.clicked:
                self.obstacle_mode = self.btn_wall.active
                self.source_mode = False
                self.btn_source.active = False

            self.tab_bar.handle_event(ev)

            if self.tab_bar.selected == 0:
                self.btn_prev.handle_event(ev)
                self.btn_next.handle_event(ev)
                self.btn_run_demo.handle_event(ev)
                if self.btn_prev.clicked:
                    self.current_lesson = max(0, self.current_lesson - 1)
                    self._highlight_sliders(LESSONS[self.current_lesson].get("highlights", []))
                if self.btn_next.clicked:
                    self.current_lesson = min(len(LESSONS) - 1, self.current_lesson + 1)
                    self._highlight_sliders(LESSONS[self.current_lesson].get("highlights", []))
                if self.btn_run_demo.clicked:
                    self._apply_preset(LESSONS[self.current_lesson]["preset"])

            if self.tab_bar.selected == 1:
                for sl in self.sliders.values():
                    sl.handle_event(ev)
                self._sync_solver_from_sliders()

            if self.tab_bar.selected == 2:
                self.cb_velocity.handle_event(ev)
                self.cb_vorticity.handle_event(ev)
                self.cb_pressure.handle_event(ev)
                self.cb_temperature.handle_event(ev)
                self.cb_colorbar.handle_event(ev)
                self.show_velocity = self.cb_velocity.checked
                self.show_vorticity = self.cb_vorticity.checked
                self.show_pressure = self.cb_pressure.checked
                self.show_temperature = self.cb_temperature.checked
                self.show_colorbar = self.cb_colorbar.checked

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self._cmap_rects:
                        for i, r in enumerate(self._cmap_rects):
                            if r.collidepoint(ev.pos):
                                self.cmap_index = i
                                self.cmap_name = self.cmap_names[i]
                    if self._grid_rects:
                        for i, r in enumerate(self._grid_rects):
                            if r.collidepoint(ev.pos):
                                self.grid_index = i
                                self.solver.resize(self.grid_options[i])
                    if self._steps_rects:
                        for i, r in enumerate(self._steps_rects):
                            if r.collidepoint(ev.pos):
                                self.steps_index = i
                                self.steps_per_frame = self.steps_options[i]

            # Canvas mouse
            rect = self._canvas_rect()
            if ev.type == pygame.MOUSEBUTTONDOWN and rect.collidepoint(ev.pos):
                self.mouse_down = ev.button
                self.last_mouse = ev.pos
                gi, gj = self._screen_to_grid(ev.pos)

                if self.obstacle_mode:
                    if ev.button == 1:
                        self.solver.set_obstacle(gi, gj, r=2, value=True)
                    elif ev.button == 3:
                        self.solver.set_obstacle(gi, gj, r=3, value=False)
                elif self.source_mode and ev.button == 1:
                    self.solver.sources.append((gi, gj, 15, 0, -4, 12))
                    self.source_mode = False
                    self.btn_source.active = False
                elif ev.button == 1:
                    self.solver.add_density(gi, gj, 80, r=4)
                    self.solver.add_velocity(gi, gj, 0, -6, r=4)
                    self.solver.add_temperature(gi, gj, 15, r=4)

            elif ev.type == pygame.MOUSEMOTION and self.mouse_down is not None:
                if rect.collidepoint(ev.pos):
                    gi, gj = self._screen_to_grid(ev.pos)
                    if self.obstacle_mode:
                        if self.mouse_down == 1:
                            self.solver.set_obstacle(gi, gj, r=2, value=True)
                        elif self.mouse_down == 3:
                            self.solver.set_obstacle(gi, gj, r=3, value=False)
                    elif self.last_mouse:
                        dx = ev.pos[0] - self.last_mouse[0]
                        dy = ev.pos[1] - self.last_mouse[1]
                        if self.mouse_down == 1:
                            self.solver.add_density(gi, gj, 40, r=3)
                            self.solver.add_velocity(gi, gj, dx * 0.5, dy * 0.5, r=3)
                            self.solver.add_temperature(gi, gj, 8, r=3)
                        elif self.mouse_down == 3:
                            self.solver.add_velocity(gi, gj, dx * 1.2, dy * 1.2, r=5)
                self.last_mouse = ev.pos

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == self.mouse_down:
                    self.mouse_down = None
                    self.last_mouse = None

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.btn_pause.active = self.paused
                    self.btn_pause.text = "Play" if self.paused else "Pause"
                elif ev.key == pygame.K_r:
                    self.solver.reset()
                elif ev.key == pygame.K_s:
                    self.solver.step()
                elif ev.key == pygame.K_v:
                    self.show_velocity = not self.show_velocity
                    self.cb_velocity.checked = self.show_velocity
                elif ev.key == pygame.K_w:
                    self.show_vorticity = not self.show_vorticity
                    self.cb_vorticity.checked = self.show_vorticity
                elif ev.key == pygame.K_p:
                    self.show_pressure = not self.show_pressure
                    self.cb_pressure.checked = self.show_pressure
                elif ev.key == pygame.K_t:
                    self.show_temperature = not self.show_temperature
                    self.cb_temperature.checked = self.show_temperature
                elif ev.key == pygame.K_LEFT:
                    self.current_lesson = max(0, self.current_lesson - 1)
                    self._highlight_sliders(LESSONS[self.current_lesson].get("highlights", []))
                elif ev.key == pygame.K_RIGHT:
                    self.current_lesson = min(len(LESSONS) - 1, self.current_lesson + 1)
                    self._highlight_sliders(LESSONS[self.current_lesson].get("highlights", []))

        return True

    # ── Main Loop ───────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            running = self._handle_events()

            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.solver.step()

            self.screen.fill(C_BASE)
            self._render_toolbar(self.screen)
            self._render_canvas(self.screen)
            self._render_panel(self.screen)
            self._render_statusbar(self.screen)
            pygame.display.flip()

            now = time.perf_counter()
            self._frame_times.append(now)
            self._frame_times = [t for t in self._frame_times if now - t < 1.0]
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = SmokeApp()
    app.run()