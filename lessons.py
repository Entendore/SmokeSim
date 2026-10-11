#!/usr/bin/env python3
"""
Educational Smoke Simulation — Lessons Edition (Pygame)
========================================================
A guided learning experience through fluid dynamics.
Side panel with 8 progressive lessons, physics sliders, and visual controls.

Install:  pip install pygame numpy scipy
Run:      python lessons.py

Controls:
  Left-drag      Smoke + Heat + Push
  Right-drag     Wind only
  Space          Pause / Play
  R              Reset simulation
  S              Single step
  V              Toggle velocity overlay
  W              Toggle vorticity overlay
  P              Toggle pressure overlay
  T              Toggle temperature overlay
  Left/Right     Navigate lessons
"""

import sys, time
import numpy as np
import pygame

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR MAPS
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
CMAP_KEYS = list(COLORMAPS.keys())

def apply_colormap(values, cmap_name):
    stops = COLORMAPS.get(cmap_name, COLORMAPS["Fire"])
    pos = [s[0] for s in stops]
    cols = np.array([s[1:] for s in stops], dtype=np.float64)
    out = np.zeros((*values.shape, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(values.ravel(), pos, cols[:, c]).reshape(values.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  FLUID SOLVER
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
#  COLOURS & FONTS
# ══════════════════════════════════════════════════════════════════════════════
C_BASE    = (30, 30, 46)
C_MANTLE  = (24, 24, 37)
C_SURF0   = (49, 50, 68)
C_SURF1   = (69, 71, 90)
C_SURF2   = (88, 91, 112)
C_TEXT    = (205, 214, 244)
C_SUB     = (166, 173, 200)
C_BLUE    = (137, 180, 250)
C_GREEN   = (166, 227, 161)
C_PEACH   = (250, 179, 135)
C_YELLOW  = (249, 226, 175)
C_RED     = (243, 139, 168)
C_TEAL    = (148, 226, 213)
C_LAVEN   = (180, 190, 254)

_FONT = None; _MONO = None

def font(size, bold=False):
    global _FONT
    if _FONT is None:
        for n in ["Segoe UI","Helvetica Neue","Arial","DejaVu Sans",None]:
            try: pygame.font.SysFont(n,12); _FONT=n; break
            except: continue
    return pygame.font.SysFont(_FONT, size, bold=bold)

def mono(size):
    global _MONO
    if _MONO is None:
        for n in ["Consolas","DejaVu Sans Mono","Courier New",None]:
            try: pygame.font.SysFont(n,12); _MONO=n; break
            except: continue
    return pygame.font.SysFont(_MONO, size)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE UI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, x, y, w, h, text, toggle=False,
                 color=C_SURF0, active_color=C_BLUE, text_color=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text; self.toggle = toggle
        self.color = color; self.active_color = active_color
        self.text_color = text_color or C_TEXT
        self.active = False; self.hovered = False; self.clicked = False

    def handle(self, ev):
        self.clicked = False
        if ev.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.clicked = True
                if self.toggle: self.active = not self.active
                return True
        return False

    def draw(self, surf):
        if self.toggle and self.active:
            col = self.active_color; tc = C_BASE
        elif self.hovered:
            col = C_SURF1; tc = self.text_color
        else:
            col = self.color; tc = self.text_color
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        pygame.draw.rect(surf, C_SURF1, self.rect, 1, border_radius=5)
        f = font(12, bold=(self.toggle and self.active))
        t = f.render(self.text, True, tc)
        surf.blit(t, (self.rect.centerx - t.get_width()//2,
                       self.rect.centery - t.get_height()//2))


class Slider:
    def __init__(self, label, lo, hi, value, highlight=False):
        self.label = label; self.lo = lo; self.hi = hi
        self.value = value; self.highlight = highlight
        self.dragging = False
        self.x = 0; self.y = 0; self.w = 280
        self._bar = pygame.Rect(0, 0, 0, 0)

    def layout(self, x, y, w):
        self.x = x; self.y = y; self.w = w
        self._bar = pygame.Rect(x + 100, y + 18, w - 140, 6)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self._bar.inflate(12, 16).collidepoint(ev.pos):
                self.dragging = True; self._set(ev.pos[0]); return True
        elif ev.type == pygame.MOUSEBUTTONUP:
            if self.dragging: self.dragging = False; return True
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set(ev.pos[0]); return True
        return False

    def _set(self, mx):
        t = max(0.0, min(1.0, (mx - self._bar.x) / max(self._bar.w, 1)))
        self.value = self.lo + t * (self.hi - self.lo)

    @property
    def iv(self): return int(round(self.value))

    def draw(self, surf):
        lc = C_YELLOW if self.highlight else C_TEXT
        surf.blit(font(12, self.highlight).render(self.label, True, lc), (self.x, self.y+2))
        surf.blit(mono(11).render(str(self.iv), True, lc), (self.x+self.w-30, self.y+2))
        pygame.draw.rect(surf, C_SURF1, self._bar, border_radius=3)
        t = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        fr = pygame.Rect(self._bar.x, self._bar.y, int(self._bar.w*t), self._bar.h)
        pygame.draw.rect(surf, C_BLUE, fr, border_radius=3)
        hx = self._bar.x + int(self._bar.w*t)
        pygame.draw.circle(surf, C_BLUE, (hx, self._bar.centery), 7)
        if self.highlight:
            pygame.draw.circle(surf, C_YELLOW, (hx, self._bar.centery), 7, 2)


class Checkbox:
    def __init__(self, label, checked=False):
        self.rect = pygame.Rect(0,0,18,18); self.label = label; self.checked = checked

    def place(self, x, y): self.rect.topleft = (x, y)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if pygame.Rect(self.rect.x, self.rect.y, 200, 20).collidepoint(ev.pos):
                self.checked = not self.checked; return True
        return False

    def draw(self, surf):
        pygame.draw.rect(surf, C_SURF0, self.rect, border_radius=3)
        pygame.draw.rect(surf, C_SURF2, self.rect, 2, border_radius=3)
        if self.checked:
            pygame.draw.rect(surf, C_BLUE, self.rect.inflate(-6,-6), border_radius=2)
        surf.blit(font(12).render(self.label, True, C_TEXT), (self.rect.right+6, self.rect.y+1))


class TabBar:
    def __init__(self, tabs):
        self.tabs = tabs; self.sel = 0; self.rects = []

    def layout(self, x, y, w):
        tw = w // len(self.tabs)
        self.rects = [pygame.Rect(x+i*tw, y, tw, 30) for i in range(len(self.tabs))]

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self.rects):
                if r.collidepoint(ev.pos): self.sel = i; return True
        return False

    def draw(self, surf):
        for i, (r, name) in enumerate(zip(self.rects, self.tabs)):
            col = C_BASE if i == self.sel else C_MANTLE
            bc = C_BLUE if i == self.sel else C_SURF0
            pygame.draw.rect(surf, col, r); pygame.draw.rect(surf, bc, r, 1)
            if i == self.sel:
                pygame.draw.line(surf, C_BLUE, (r.x+1,r.bottom-2), (r.right-1,r.bottom-2), 2)
            tc = C_TEXT if i == self.sel else C_SUB
            t = font(12, i == self.sel).render(name, True, tc)
            surf.blit(t, (r.centerx-t.get_width()//2, r.centery-t.get_height()//2))


# ══════════════════════════════════════════════════════════════════════════════
#  LESSONS DATA
# ══════════════════════════════════════════════════════════════════════════════
LESSONS = [
    {
        "title": "1. What is Fluid Sim?",
        "preset": "candle",
        "highlights": ["buoyancy"],
        "lines": [
            ("h", "What is Fluid Simulation?"),
            ("", ""),
            ("", "Solving the Navier-Stokes equations --"),
            ("", "  governing weather, oceans, and smoke."),
            ("", ""),
            ("", "Why is it hard?"),
            ("", "  Non-linear + incompressibility constraint."),
            ("", ""),
            ("", "Stam's Insight: Trade accuracy for"),
            ("", "  guaranteed stability."),
            ("", ""),
            ("g", ">> Click 'Run This Demo' for a rising plume."),
            ("g", ">> Drag mouse on canvas to interact!"),
        ],
    },
    {
        "title": "2. Navier-Stokes Eqns",
        "preset": "default",
        "highlights": ["viscosity"],
        "lines": [
            ("h", "Navier-Stokes Equations"),
            ("", ""),
            ("e", "  du/dt = -(u.grad)u + v*Lap(u) - grad(p) + f"),
            ("", ""),
            ("k", "  -(u.grad)u      Advection"),
            ("k", "  v*Lap(u)        Viscous Diffusion"),
            ("k", "  -grad(p)        Pressure"),
            ("k", "  f               External Forces"),
            ("", ""),
            ("e", "  div(u) = 0      (Incompressibility)"),
            ("", ""),
            ("", "These partial differential equations"),
            ("", "describe how fluids move. Each term"),
            ("", "captures a different physical effect."),
        ],
    },
    {
        "title": "3. Advection",
        "preset": "advection",
        "highlights": ["dt"],
        "lines": [
            ("h", "Advection"),
            ("", ""),
            ("", "Carrying quantities along the velocity"),
            ("", "field. Think of leaves on a river."),
            ("", ""),
            ("r", "Naive:  x_new = x + dt*u(x)"),
            ("r", "  >> Unstable for large dt!"),
            ("", ""),
            ("g", "Semi-Lagrangian (Stam 1999):"),
            ("g", "  Trace backwards from each cell."),
            ("g", "  Sample the old field via bilinear"),
            ("g", "  interpolation.  >> Unconditionally stable!"),
            ("", ""),
            ("", "Notice: diffusion is set to 0 here"),
            ("", "so you can see pure advection behavior."),
        ],
    },
    {
        "title": "4. Diffusion",
        "preset": "diffusion",
        "highlights": ["viscosity", "diffusion"],
        "lines": [
            ("h", "Diffusion"),
            ("", ""),
            ("", "Velocity spreads via viscosity (v)."),
            ("", "Density spreads via diffusion (k)."),
            ("", ""),
            ("", "Higher values = more blurry/smooth."),
            ("", "Try dragging the Diffusion slider up!"),
            ("", ""),
            ("g", "Implicit Solve: Gauss-Seidel relaxation."),
            ("g", "  >> Unconditionally stable, unlike"),
            ("g", "     explicit methods that explode."),
            ("", ""),
            ("", "The grayscale colormap makes the"),
            ("", "diffusion spreading effect very clear."),
        ],
    },
    {
        "title": "5. Pressure",
        "preset": "pressure",
        "highlights": [],
        "lines": [
            ("h", "Pressure Projection"),
            ("", ""),
            ("", "After advection + diffusion, the velocity"),
            ("", "field may no longer be divergence-free."),
            ("", ""),
            ("", "Helmholtz-Hodge Decomposition fixes this:"),
            ("k", "  1. Compute div(u)"),
            ("k", "  2. Solve Poisson: Lap(p) = div(u)"),
            ("k", "  3. Subtract grad(p) from u"),
            ("", ""),
            ("g", ">> Pressure overlay is ON -- watch the"),
            ("g", "   red/blue pressure field respond to"),
            ("g", "   the smoke plume in real time."),
        ],
    },
    {
        "title": "6. Forces",
        "preset": "vorticity",
        "highlights": ["buoyancy", "vorticity"],
        "lines": [
            ("h", "External Forces"),
            ("", ""),
            ("k", "Buoyancy:"),
            ("", "  Temperature drives upward force."),
            ("", "  Hot fluid rises, cold fluid sinks."),
            ("", "  Try the buoyancy slider: 0 vs 20!"),
            ("", ""),
            ("k", "Vorticity Confinement:"),
            ("", "  Numerical diffusion kills small-scale"),
            ("", "  swirls. This force re-injects lost"),
            ("", "  rotational energy back into the flow."),
            ("", ""),
            ("g", ">> Vorticity overlay ON. Set vorticity"),
            ("g", "   to 0 to see the difference!"),
        ],
    },
    {
        "title": "7. Obstacles",
        "preset": "obstacle",
        "highlights": [],
        "lines": [
            ("h", "Obstacles"),
            ("", ""),
            ("", "Walls enforce zero velocity inside them"),
            ("", "and alter the surrounding pressure field."),
            ("", ""),
            ("", "The solver treats obstacle cells specially"),
            ("", "during boundary conditions and projection."),
            ("", ""),
            ("g", ">> A wall with a gap is pre-placed."),
            ("g", "   Watch smoke squeeze through!"),
            ("", ""),
            ("", "Toggle Wall Mode (toolbar or W key):"),
            ("k", "  Left-drag  =  draw walls"),
            ("k", "  Right-drag =  erase walls"),
        ],
    },
    {
        "title": "8. Playground",
        "preset": "playground",
        "highlights": [],
        "lines": [
            ("h", "Playground"),
            ("", ""),
            ("", "You've learned all the building blocks!"),
            ("", "Now experiment freely."),
            ("", ""),
            ("k", "Left-drag   Smoke + Heat + Push"),
            ("k", "Right-drag  Wind only"),
            ("", ""),
            ("k", "Keyboard:"),
            ("k", "  Space   Pause / Play"),
            ("k", "  R       Reset"),
            ("k", "  S       Single step"),
            ("k", "  V/W/P/T Toggle overlays"),
            ("k", "  Left/Right  Switch lessons"),
            ("", ""),
            ("g", "Have fun exploring fluid dynamics!"),
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LESSONS APP
# ══════════════════════════════════════════════════════════════════════════════
class LessonsApp:
    PANEL_W = 340
    TB_H = 42
    SB_H = 24

    def __init__(self):
        pygame.init()
        self.W, self.H = 1280, 800
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("Fluid Simulation Lessons")
        self.clock = pygame.time.Clock()

        self.solver = FluidSolver(N=100)
        self.paused = False
        self.cmap_idx = 0
        self.show_vel = False; self.show_vor = False
        self.show_prs = False; self.show_tmp = False; self.show_bar = True
        self.obstacle_mode = False; self.source_mode = False
        self.steps_per_frame = 1
        self.lesson_idx = 0

        self.mouse_down = None; self.last_mouse = None
        self._ft = []

        # Widgets
        self.tabs = TabBar(["Learn", "Physics", "Visuals"])
        self.btn_pause = Button(8, 6, 80, 30, "Pause", toggle=True)
        self.btn_step  = Button(96, 6, 66, 30, "Step")
        self.btn_reset = Button(170, 6, 76, 30, "Reset")
        self.btn_source = Button(256, 6, 80, 30, "Source", toggle=True)
        self.btn_wall   = Button(344, 6, 66, 30, "Wall", toggle=True)
        self.tb_btns = [self.btn_pause, self.btn_step, self.btn_reset,
                        self.btn_source, self.btn_wall]
        self.btn_prev = Button(0, 0, 32, 30, "<")
        self.btn_next = Button(0, 0, 32, 30, ">")
        self.btn_demo = Button(0, 0, 300, 34, "Run This Demo",
                               color=(30,102,64), active_color=(42,128,80))

        self.sliders = {
            "viscosity": Slider("Viscosity (v)",    0, 50,  1),
            "diffusion": Slider("Diffusion (k)",    0, 50,  1),
            "buoyancy":  Slider("Buoyancy (a)",     0, 100, 10),
            "vorticity": Slider("Vorticity (e)",    0, 250, 50),
            "dt":        Slider("Time Step",        1, 30,  10),
            "decay":     Slider("Decay (permille)", 980,1000,997),
        }
        self.cb_vel = Checkbox("Velocity", False)
        self.cb_vor = Checkbox("Vorticity", False)
        self.cb_prs = Checkbox("Pressure", False)
        self.cb_tmp = Checkbox("Temp", False)
        self.cb_bar = Checkbox("Colour bar", True)
        self.grid_opts = [64, 100, 128, 160]; self.grid_idx = 1
        self.steps_opts = list(range(1,9)); self.steps_idx = 0
        self._cmap_rects = []; self._grid_rects = []; self._steps_rects = []

        self._apply_preset("candle")

    @property
    def cmap_name(self): return CMAP_KEYS[self.cmap_idx]

    # ── Canvas ──────────────────────────────────────────────────────────
    def _crect(self):
        cw = self.W - self.PANEL_W - 30; ch = self.H - self.TB_H - self.SB_H - 20
        s = min(cw, ch)
        return pygame.Rect(10+(cw-s)//2, self.TB_H+10+(ch-s)//2, s, s)

    def _s2g(self, pos):
        r = self._crect(); N = self.solver.N
        return (int(np.clip((pos[0]-r.x)/r.w*(N+2),1,N)),
                int(np.clip((pos[1]-r.y)/r.h*(N+2),1,N)))

    # ── Presets ─────────────────────────────────────────────────────────
    def _apply_preset(self, name):
        self.solver.reset(); N = self.solver.N; cx = N//2
        P = {
            "candle":    dict(buoy=2.0,vort=3.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,
                              src=[(cx,N-5,25,0,-6,12)],cmap="Fire"),
            "default":   dict(buoy=1.0,vort=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,
                              src=[],cmap="Fire"),
            "advection": dict(buoy=1.0,vort=5.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,
                              src=[(cx,N-5,20,0,-5,10)],cmap="Cool"),
            "diffusion": dict(buoy=1.5,vort=0.0,visc=0.005,diff=0.005,decay=0.997,dt=0.1,
                              src=[(cx,N-5,25,0,-5,12)],cmap="Grayscale"),
            "pressure":  dict(buoy=2.0,vort=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,
                              src=[(cx,N-5,25,0,-6,12)],cmap="Fire",ov={"pressure":True}),
            "vorticity": dict(buoy=2.0,vort=20.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,
                              src=[(cx,N-5,25,0,-5,12)],cmap="Neon",ov={"vorticity":True}),
            "obstacle":  dict(buoy=2.0,vort=8.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,
                              src=[(cx,N-5,30,0,-7,15)],cmap="Thermal",obs="wall"),
            "playground":dict(buoy=1.0,vort=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,
                              src=[],cmap="Fire"),
        }
        p = P.get(name, P["default"])
        s = self.solver
        s.buoyancy=p["buoy"]; s.vort_conf=p["vort"]; s.visc=p["visc"]
        s.diff=p["diff"]; s.decay=p["decay"]; s.dt=p["dt"]
        for src in p.get("src",[]): s.sources.append(src)
        if p.get("obs") == "wall":
            for i in range(1,N+1):
                for j in range(N//2-2, N//2+2):
                    if abs(i-cx) > 6: s.obstacle[i,j] = True
        self.show_prs = p.get("ov",{}).get("pressure", False)
        self.show_vor = p.get("ov",{}).get("vorticity", False)
        self.cb_prs.checked = self.show_prs; self.cb_vor.checked = self.show_vor
        cn = p.get("cmap","Fire")
        self.cmap_idx = CMAP_KEYS.index(cn) if cn in CMAP_KEYS else 0
        self._sync_sl_from_solver()
        self._highlight(LESSONS[self.lesson_idx].get("highlights", []))

    def _sync_sl_from_solver(self):
        s = self.solver
        self.sliders["viscosity"].value = int(s.visc/1e-4)
        self.sliders["diffusion"].value = int(s.diff/1e-4)
        self.sliders["buoyancy"].value = int(s.buoyancy*10)
        self.sliders["vorticity"].value = int(s.vort_conf*10)
        self.sliders["dt"].value = int(s.dt*100)
        self.sliders["decay"].value = int(s.decay*1000)

    def _sync_solver(self):
        s = self.solver
        s.visc = self.sliders["viscosity"].value*1e-4
        s.diff = self.sliders["diffusion"].value*1e-4
        s.buoyancy = self.sliders["buoyancy"].value/10.0
        s.vort_conf = self.sliders["vorticity"].value/10.0
        s.dt = self.sliders["dt"].value/100.0
        s.decay = self.sliders["decay"].value/1000.0

    def _highlight(self, keys):
        m = {"buoyancy":"buoyancy","viscosity":"viscosity","vorticity":"vorticity","diffusion":"diffusion"}
        active = set(m.get(k,k) for k in keys)
        for k, sl in self.sliders.items(): sl.highlight = k in active

    # ── Canvas Rendering ────────────────────────────────────────────────
    def _render_canvas(self, surf):
        rect = self._crect(); N = self.solver.N
        pygame.draw.rect(surf, (8,8,14), rect, border_radius=4)
        pygame.draw.rect(surf, C_SURF0, rect.inflate(2,2), 1, border_radius=5)

        d = self.solver.d[1:N+1,1:N+1].T; dmax = max(d.max(), 1e-6)
        dn = np.clip(d/dmax, 0, 1)**0.6
        rgb = np.ascontiguousarray(apply_colormap(dn, self.cmap_name))
        gs = pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
        gs = pygame.transform.smoothscale(gs, (rect.w, rect.h))
        surf.blit(gs, rect.topleft)

        if self.show_tmp: self._overlay(surf, rect, N, self._tmp_rgb, 0.35)
        if self.show_prs: self._overlay(surf, rect, N, self._prs_rgb, 0.30)
        if self.show_vor: self._overlay(surf, rect, N, self._vor_rgb, 0.35)
        self._obstacles(surf, rect, N)
        if self.show_vel: self._arrows(surf, rect, N)

        for sx, sy, *_ in self.solver.sources:
            mx = rect.x+sx/(N+2)*rect.w; my = rect.y+sy/(N+2)*rect.h
            pygame.draw.circle(surf, (255,220,50), (int(mx),int(my)), 7, 2)
            pygame.draw.line(surf, (255,220,50), (int(mx)-4,int(my)), (int(mx)+4,int(my)), 1)
            pygame.draw.line(surf, (255,220,50), (int(mx),int(my)-4), (int(mx),int(my)+4), 1)

        if self.source_mode:
            t = font(13,True).render("Source Mode -- click canvas to place", True, C_YELLOW)
            surf.blit(t, (rect.centerx-t.get_width()//2, rect.y+8))
        elif self.obstacle_mode:
            t = font(13,True).render("Wall Mode (R-click: Erase)", True, C_RED)
            surf.blit(t, (rect.centerx-t.get_width()//2, rect.y+8))
        if self.show_bar: self._colorbar(surf, rect, dmax)

    def _overlay(self, surf, rect, N, fn, a):
        rgb = fn(N)
        o = pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
        o = pygame.transform.smoothscale(o, (rect.w, rect.h))
        o.set_alpha(int(a*255)); surf.blit(o, rect.topleft)

    def _tmp_rgb(self, N):
        T = self.solver.T[1:N+1,1:N+1].T; tm = max(T.max(),1e-6); tn = np.clip(T/tm,0,1)
        return np.stack([(tn*220+20).astype(np.uint8),(tn*120+10).astype(np.uint8),
                         np.full(N*N,15,dtype=np.uint8).reshape(N,N)], axis=-1)

    def _prs_rgb(self, N):
        pf = self.solver.debug.get('pressure')
        if pf is None: return np.zeros((N,N,3),dtype=np.uint8)
        d = pf[1:N+1,1:N+1].T; mx = max(np.abs(d).max(),1e-8); dn=d/mx
        p,n = np.clip(dn,0,1), np.clip(-dn,0,1)
        return np.stack([(p*200+30).astype(np.uint8),np.full(N*N,30,dtype=np.uint8).reshape(N,N),
                         (n*200+30).astype(np.uint8)], axis=-1)

    def _vor_rgb(self, N):
        wf = self.solver.debug.get('vorticity')
        if wf is None: return np.zeros((N,N,3),dtype=np.uint8)
        w = wf[1:N+1,1:N+1].T; wn = np.clip(w/max(np.abs(w).max(),1e-8),-1,1)
        p,n = np.clip(wn,0,1), np.clip(-wn,0,1)
        return np.stack([(p*220+20).astype(np.uint8),np.full(N*N,20,dtype=np.uint8).reshape(N,N),
                         (n*220+20).astype(np.uint8)], axis=-1)

    def _obstacles(self, surf, rect, N):
        obs = self.solver.obstacle[1:N+1,1:N+1].T
        if not obs.any(): return
        rgba = np.zeros((N,N,4),dtype=np.uint8)
        rgba[obs,0]=55;rgba[obs,1]=60;rgba[obs,2]=75;rgba[obs,3]=220
        try:
            from scipy.ndimage import binary_dilation
            edge = binary_dilation(obs,iterations=1)&~obs
            if edge.any(): rgba[edge,0]=100;rgba[edge,1]=110;rgba[edge,2]=140;rgba[edge,3]=180
        except ImportError: pass
        pix = np.zeros((N,N,4),dtype=np.uint8)
        pix[:,:,0:3]=np.transpose(rgba[:,:,0:3],(1,0,2)); pix[:,:,3]=rgba[:,:,3].T
        s = pygame.image.frombuffer(pix.tobytes(),(N,N),"RGBA").convert_alpha()
        surf.blit(pygame.transform.smoothscale(s,(rect.w,rect.h)), rect.topleft)

    def _arrows(self, surf, rect, N):
        step = max(N//22,1); vmax = max(self.solver.max_velocity(),0.1)
        sc = min(rect.w,rect.h)/(N+2)*0.8/vmax*step; sc = max(0.5,min(sc,30))
        for i in range(1,N+1,step):
            for j in range(1,N+1,step):
                if self.solver.obstacle[i,j]: continue
                ux,vy = self.solver.u[i,j],self.solver.v[i,j]
                if abs(ux)+abs(vy)<vmax*0.01: continue
                sx=rect.x+i/(N+2)*rect.w; sy=rect.y+j/(N+2)*rect.h
                pygame.draw.line(surf,(130,200,255),(int(sx),int(sy)),(int(sx+ux*sc),int(sy+vy*sc)),1)

    def _colorbar(self, surf, rect, dmax):
        cw,ch=16,100; x0=rect.right-cw-12; y0=rect.bottom-ch-24
        bg=pygame.Surface((cw+48,ch+24),pygame.SRCALPHA); bg.fill((20,20,35,160))
        surf.blit(bg,(x0-6,y0-14))
        for i in range(ch):
            t=1.0-i/ch; rgb=apply_colormap(np.array([[t]]),self.cmap_name)[0,0]
            pygame.draw.line(surf,(int(rgb[0]),int(rgb[1]),int(rgb[2])),(x0,y0+i),(x0+cw,y0+i))
        pygame.draw.rect(surf,C_SUB,(x0-1,y0-1,cw+2,ch+2),1)
        mf=mono(9)
        surf.blit(mf.render(f"{dmax:.0f}",True,C_SUB),(x0+cw+3,y0))
        surf.blit(mf.render("0",True,C_SUB),(x0+cw+3,y0+ch-9))

    # ── Toolbar / Status / Panel ────────────────────────────────────────
    def _draw_toolbar(self, surf):
        pygame.draw.rect(surf, C_MANTLE, (0,0,self.W,self.TB_H))
        pygame.draw.line(surf, C_SURF0, (0,self.TB_H-1),(self.W,self.TB_H-1))
        for b in self.tb_btns: b.draw(surf)
        lbl = font(12,True).render(LESSONS[self.lesson_idx]["title"], True, C_GREEN)
        surf.blit(lbl, (430, 14))

    def _draw_status(self, surf):
        y = self.H-self.SB_H
        pygame.draw.rect(surf, C_MANTLE, (0,y,self.W,self.SB_H))
        pygame.draw.line(surf, C_SURF0, (0,y),(self.W,y))
        s = self.solver; fps = len(self._ft)
        t = mono(11).render(
            f"FPS:{fps}  Solve:{s.solve_ms:.1f}ms  Grid:{s.N}^2  Step:{s.step_count}  "
            f"MaxV:{s.max_velocity():.1f}  SumD:{s.total_density():.0f}  Src:{len(s.sources)}",
            True, C_SUB)
        surf.blit(t, (10, y+5))

    def _draw_panel(self, surf):
        px = self.W - self.PANEL_W
        pr = pygame.Rect(px, self.TB_H, self.PANEL_W, self.H-self.TB_H-self.SB_H)
        pygame.draw.rect(surf, C_BASE, pr)
        pygame.draw.line(surf, C_SURF0, (px,self.TB_H), (px,self.H-self.SB_H), 2)
        self.tabs.layout(px, self.TB_H, self.PANEL_W)
        self.tabs.draw(surf)
        ty = self.TB_H + 36
        if self.tabs.sel == 0:   self._panel_learn(surf, px, ty)
        elif self.tabs.sel == 1: self._panel_physics(surf, px, ty)
        elif self.tabs.sel == 2: self._panel_visuals(surf, px, ty)

    def _panel_learn(self, surf, px, y):
        self.btn_prev.rect.topleft = (px+12, y); self.btn_prev.draw(surf)
        self.btn_next.rect.topleft = (px+self.PANEL_W-44, y); self.btn_next.draw(surf)
        t = font(12,True).render(LESSONS[self.lesson_idx]["title"], True, C_BLUE)
        surf.blit(t, (px+52, y+7))
        # Text area
        ta = pygame.Rect(px+8, y+38, self.PANEL_W-16, self.H-self.SB_H-y-80)
        pygame.draw.rect(surf, C_MANTLE, ta, border_radius=6)
        pygame.draw.rect(surf, C_SURF0, ta, 1, border_radius=6)
        lesson = LESSONS[self.lesson_idx]
        ly = ta.y + 12
        for tag, line in lesson["lines"]:
            if tag == "h":   c, f = C_PEACH, font(14, True)
            elif tag == "g": c, f = C_GREEN, font(12)
            elif tag == "r": c, f = C_RED, font(12)
            elif tag == "k": c, f = C_BLUE, font(12)
            elif tag == "e": c, f = C_TEAL, mono(13)
            else:            c, f = C_TEXT, font(12)
            surf.blit(f.render(line, True, c), (ta.x+14, ly))
            ly += 20
        # Demo button
        self.btn_demo.rect = pygame.Rect(px+12, ta.bottom+8, self.PANEL_W-24, 34)
        self.btn_demo.draw(surf)
        # Progress dots
        dot_y = ta.bottom + 50
        total = len(LESSONS)
        dot_w = total * 16
        dot_x = px + (self.PANEL_W - dot_w) // 2
        for i in range(total):
            c = C_BLUE if i == self.lesson_idx else C_SURF1
            pygame.draw.circle(surf, c, (dot_x + i*16 + 8, dot_y + 6), 4)

    def _panel_physics(self, surf, px, y):
        surf.blit(font(13,True).render("Parameters", True, C_BLUE), (px+14, y+4))
        sy = y + 30
        for key in ["viscosity","diffusion","buoyancy","vorticity","dt","decay"]:
            sl = self.sliders[key]; sl.layout(px+12, sy, self.PANEL_W-24); sl.draw(surf)
            sy += 48
        # Hint
        surf.blit(font(11).render("Highlighted sliders relate to", True, C_SUB), (px+14, sy+8))
        surf.blit(font(11).render("the current lesson topic.", True, C_SUB), (px+14, sy+24))

    def _panel_visuals(self, surf, px, y):
        # Colour maps
        surf.blit(font(13).render("Colour Map:", True, C_TEXT), (px+14, y+6))
        cy = y + 26; self._cmap_rects = []
        bw = (self.PANEL_W-40)//3
        for i, name in enumerate(CMAP_KEYS):
            row, col = divmod(i, 3)
            r = pygame.Rect(px+12+col*(bw+6), cy+row*32, bw, 26)
            self._cmap_rects.append(r)
            if i == self.cmap_idx:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=5); tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=5)
                pygame.draw.rect(surf, C_SURF1, r, 1, border_radius=5); tc = C_TEXT
            t = font(11).render(name, True, tc)
            surf.blit(t, (r.centerx-t.get_width()//2, r.centery-t.get_height()//2))

        # Overlays
        oy = cy + 80
        surf.blit(font(13,True).render("Overlays", True, C_BLUE), (px+14, oy)); oy += 26
        self.cb_vel.place(px+16, oy); self.cb_vor.place(px+146, oy)
        self.cb_vel.draw(surf); self.cb_vor.draw(surf); oy += 26
        self.cb_prs.place(px+16, oy); self.cb_tmp.place(px+146, oy)
        self.cb_prs.draw(surf); self.cb_tmp.draw(surf); oy += 26
        self.cb_bar.place(px+16, oy); self.cb_bar.draw(surf); oy += 36

        # Grid size
        surf.blit(font(13,True).render("Grid Size:", True, C_BLUE), (px+14, oy)); oy += 24
        self._grid_rects = []; gw = 58
        for i, g in enumerate(self.grid_opts):
            r = pygame.Rect(px+16+i*(gw+6), oy, gw, 26); self._grid_rects.append(r)
            if i == self.grid_idx:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=4); tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=4); tc = C_TEXT
            t = font(11).render(str(g), True, tc)
            surf.blit(t, (r.centerx-t.get_width()//2, r.centery-t.get_height()//2))
        oy += 40

        # Steps/frame
        surf.blit(font(13,True).render("Steps / Frame:", True, C_BLUE), (px+14, oy)); oy += 24
        self._steps_rects = []; sw = 32
        for i, s in enumerate(self.steps_opts):
            r = pygame.Rect(px+16+i*(sw+4), oy, sw, 24); self._steps_rects.append(r)
            if i == self.steps_idx:
                pygame.draw.rect(surf, C_BLUE, r, border_radius=4); tc = C_BASE
            else:
                pygame.draw.rect(surf, C_SURF0, r, border_radius=4); tc = C_TEXT
            t = font(11).render(str(s), True, tc)
            surf.blit(t, (r.centerx-t.get_width()//2, r.centery-t.get_height()//2))

    # ── Events ──────────────────────────────────────────────────────────
    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            if ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)

            for b in self.tb_btns: b.handle(ev)
            if self.btn_pause.clicked:
                self.paused = self.btn_pause.active
                self.btn_pause.text = "Play" if self.paused else "Pause"
            if self.btn_step.clicked: self.solver.step()
            if self.btn_reset.clicked: self.solver.reset()
            if self.btn_source.clicked:
                self.source_mode = self.btn_source.active
                self.obstacle_mode = False; self.btn_wall.active = False
            if self.btn_wall.clicked:
                self.obstacle_mode = self.btn_wall.active
                self.source_mode = False; self.btn_source.active = False

            self.tabs.handle(ev)

            # Learn tab
            if self.tabs.sel == 0:
                self.btn_prev.handle(ev); self.btn_next.handle(ev); self.btn_demo.handle(ev)
                if self.btn_prev.clicked:
                    self.lesson_idx = max(0, self.lesson_idx-1)
                    self._highlight(LESSONS[self.lesson_idx].get("highlights",[]))
                if self.btn_next.clicked:
                    self.lesson_idx = min(len(LESSONS)-1, self.lesson_idx+1)
                    self._highlight(LESSONS[self.lesson_idx].get("highlights",[]))
                if self.btn_demo.clicked:
                    self._apply_preset(LESSONS[self.lesson_idx]["preset"])

            # Physics tab
            if self.tabs.sel == 1:
                for sl in self.sliders.values(): sl.handle(ev)
                self._sync_solver()

            # Visuals tab
            if self.tabs.sel == 2:
                for cb in [self.cb_vel, self.cb_vor, self.cb_prs, self.cb_tmp, self.cb_bar]:
                    cb.handle(ev)
                self.show_vel = self.cb_vel.checked; self.show_vor = self.cb_vor.checked
                self.show_prs = self.cb_prs.checked; self.show_tmp = self.cb_tmp.checked
                self.show_bar = self.cb_bar.checked
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for i, r in enumerate(self._cmap_rects):
                        if r.collidepoint(ev.pos): self.cmap_idx = i
                    for i, r in enumerate(self._grid_rects):
                        if r.collidepoint(ev.pos):
                            self.grid_idx = i; self.solver.resize(self.grid_opts[i])
                    for i, r in enumerate(self._steps_rects):
                        if r.collidepoint(ev.pos):
                            self.steps_idx = i; self.steps_per_frame = self.steps_opts[i]

            # Canvas mouse
            rect = self._crect()
            if ev.type == pygame.MOUSEBUTTONDOWN and rect.collidepoint(ev.pos):
                self.mouse_down = ev.button; self.last_mouse = ev.pos
                gi, gj = self._s2g(ev.pos)
                if self.obstacle_mode:
                    if ev.button == 1: self.solver.set_obstacle(gi,gj,r=2,value=True)
                    elif ev.button == 3: self.solver.set_obstacle(gi,gj,r=3,value=False)
                elif self.source_mode and ev.button == 1:
                    self.solver.sources.append((gi,gj,15,0,-4,12))
                    self.source_mode = False; self.btn_source.active = False
                elif ev.button == 1:
                    self.solver.add_density(gi,gj,80,r=4)
                    self.solver.add_velocity(gi,gj,0,-6,r=4)
                    self.solver.add_temperature(gi,gj,15,r=4)
            elif ev.type == pygame.MOUSEMOTION and self.mouse_down is not None:
                if rect.collidepoint(ev.pos):
                    gi, gj = self._s2g(ev.pos)
                    if self.obstacle_mode:
                        if self.mouse_down == 1: self.solver.set_obstacle(gi,gj,r=2,value=True)
                        elif self.mouse_down == 3: self.solver.set_obstacle(gi,gj,r=3,value=False)
                    elif self.last_mouse:
                        dx = ev.pos[0]-self.last_mouse[0]; dy = ev.pos[1]-self.last_mouse[1]
                        if self.mouse_down == 1:
                            self.solver.add_density(gi,gj,40,r=3)
                            self.solver.add_velocity(gi,gj,dx*0.5,dy*0.5,r=3)
                            self.solver.add_temperature(gi,gj,8,r=3)
                        elif self.mouse_down == 3:
                            self.solver.add_velocity(gi,gj,dx*1.2,dy*1.2,r=5)
                self.last_mouse = ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == self.mouse_down:
                    self.mouse_down = None; self.last_mouse = None

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.btn_pause.active = self.paused
                    self.btn_pause.text = "Play" if self.paused else "Pause"
                elif ev.key == pygame.K_r: self.solver.reset()
                elif ev.key == pygame.K_s: self.solver.step()
                elif ev.key == pygame.K_v: self.show_vel=not self.show_vel; self.cb_vel.checked=self.show_vel
                elif ev.key == pygame.K_w: self.show_vor=not self.show_vor; self.cb_vor.checked=self.show_vor
                elif ev.key == pygame.K_p: self.show_prs=not self.show_prs; self.cb_prs.checked=self.show_prs
                elif ev.key == pygame.K_t: self.show_tmp=not self.show_tmp; self.cb_tmp.checked=self.show_tmp
                elif ev.key == pygame.K_LEFT:
                    self.lesson_idx = max(0, self.lesson_idx-1)
                    self._highlight(LESSONS[self.lesson_idx].get("highlights",[]))
                elif ev.key == pygame.K_RIGHT:
                    self.lesson_idx = min(len(LESSONS)-1, self.lesson_idx+1)
                    self._highlight(LESSONS[self.lesson_idx].get("highlights",[]))
        return True

    # ── Main Loop ───────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            running = self._events()
            if not self.paused:
                for _ in range(self.steps_per_frame): self.solver.step()
            self.screen.fill(C_BASE)
            self._draw_toolbar(self.screen)
            self._render_canvas(self.screen)
            self._draw_panel(self.screen)
            self._draw_status(self.screen)
            pygame.display.flip()
            now = time.perf_counter()
            self._ft.append(now); self._ft = [t for t in self._ft if now-t<1.0]
            self.clock.tick(60)
        pygame.quit(); sys.exit()


if __name__ == "__main__":
    LessonsApp().run()