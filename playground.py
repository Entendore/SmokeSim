#!/usr/bin/env python3
"""
Smoke Simulation Playground — Pygame
=====================================
A minimal, full-screen fluid sandbox. No lessons, no side panel —
just the simulation filling your window with a compact floating toolbar.

Install:  pip install pygame numpy scipy
Run:      python playground.py

Controls:
  Left-drag      Smoke + Heat + Push
  Right-drag     Wind only
  Middle-click   Place permanent emitter
  Space          Pause / Play
  R              Reset
  S              Single step
  V              Toggle velocity arrows
  W              Toggle vorticity overlay
  P              Toggle pressure overlay
  T              Toggle temperature overlay
  B              Toggle colour bar
  1-6            Switch colour map
  [ / ]          Decrease / Increase grid size
  Scroll wheel   Adjust buoyancy
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
    def __init__(self, N=128):
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
C_TEXT    = (205, 214, 244)
C_SUB     = (166, 173, 200)
C_BLUE    = (137, 180, 250)
C_GREEN   = (166, 227, 161)
C_PEACH   = (250, 179, 135)
C_RED     = (243, 139, 168)
C_YELLOW  = (249, 226, 175)

_FONT = None
_MONO = None
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
#  PLAYGROUND APP
# ══════════════════════════════════════════════════════════════════════════════
class Playground:
    TOOLBAR_H = 44
    STATUS_H  = 24
    MARGIN     = 12

    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.W = min(1400, info.current_w - 80)
        self.H = min(900, info.current_h - 80)
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("Smoke Playground")
        self.clock = pygame.time.Clock()

        self.solver = FluidSolver(N=128)
        self.paused = False
        self.cmap_idx = 0
        self.show_vel = False
        self.show_vor = False
        self.show_prs = False
        self.show_tmp = False
        self.show_bar = True
        self.obstacle_mode = False
        self.steps_per_frame = 1

        self.mouse_down = None
        self.last_mouse = None
        self._frame_times = []
        self._toolbar_buttons = []  # rebuilt each frame

    @property
    def cmap_name(self):
        return CMAP_KEYS[self.cmap_idx]

    # ── Canvas rect (fills available space) ─────────────────────────────
    def _canvas_rect(self):
        m = self.MARGIN
        return pygame.Rect(m, self.TOOLBAR_H + m,
                           self.W - 2*m, self.H - self.TOOLBAR_H - self.STATUS_H - 2*m)

    def _screen_to_grid(self, pos):
        r = self._canvas_rect(); N = self.solver.N
        gi = int(np.clip((pos[0]-r.x)/r.w*(N+2), 1, N))
        gj = int(np.clip((pos[1]-r.y)/r.h*(N+2), 1, N))
        return gi, gj

    # ── Floating toolbar ────────────────────────────────────────────────
    def _build_toolbar_buttons(self):
        """Return list of (rect, label, is_active) for the top toolbar."""
        btns = []
        x = 12; y = 7; h = 30; gap = 6
        def add(w, label, active=False):
            nonlocal x
            btns.append((pygame.Rect(x, y, w, h), label, active))
            x += w + gap

        add(76, "Pause" if not self.paused else "Play", self.paused)
        add(66, "Reset")
        add(66, "Wall", self.obstacle_mode)
        x += gap  # extra gap
        add(62, self.cmap_name)
        x += gap
        add(30, "Vel", self.show_vel)
        add(36, "Vort", self.show_vor)
        add(36, "Pres", self.show_prs)
        add(36, "Temp", self.show_tmp)
        add(36, "Bar", self.show_bar)
        x += gap
        add(80, f"Grid {self.solver.N}")
        add(50, f"x{self.steps_per_frame}")
        return btns

    def _handle_toolbar_click(self, pos, btns):
        for i, (rect, label, _) in enumerate(btns):
            if not rect.collidepoint(pos):
                continue
            if i == 0:
                self.paused = not self.paused
            elif i == 1:
                self.solver.reset()
            elif i == 2:
                self.obstacle_mode = not self.obstacle_mode
            elif i == 3:
                self.cmap_idx = (self.cmap_idx + 1) % len(CMAP_KEYS)
            elif i == 4:
                self.show_vel = not self.show_vel
            elif i == 5:
                self.show_vor = not self.show_vor
            elif i == 6:
                self.show_prs = not self.show_prs
            elif i == 7:
                self.show_tmp = not self.show_tmp
            elif i == 8:
                self.show_bar = not self.show_bar
            elif i == 9:
                sizes = [64, 80, 100, 128, 160, 200]
                cur = self.solver.N
                try:
                    idx = sizes.index(cur)
                    idx = (idx + 1) % len(sizes)
                except ValueError:
                    idx = 3
                self.solver.resize(sizes[idx])
            elif i == 10:
                self.steps_per_frame = self.steps_per_frame % 8 + 1
            return True
        return False

    def _draw_toolbar(self, surf, btns):
        pygame.draw.rect(surf, C_MANTLE, (0, 0, self.W, self.TOOLBAR_H))
        pygame.draw.line(surf, C_SURF0, (0, self.TOOLBAR_H-1), (self.W, self.TOOLBAR_H-1))
        for rect, label, active in btns:
            if active:
                col, tc = C_BLUE, C_MANTLE
            else:
                col, tc = C_SURF0, C_TEXT
            pygame.draw.rect(surf, col, rect, border_radius=5)
            pygame.draw.rect(surf, C_SURF1, rect, 1, border_radius=5)
            txt = font(12, bold=active).render(label, True, tc)
            surf.blit(txt, (rect.centerx - txt.get_width()//2,
                            rect.centery - txt.get_height()//2))
        # Key hints on the right
        hints = font(10).render(
            "L-drag:smoke  R-drag:wind  M-click:emitter  Space:pause  R:reset  1-6:colormap  [/]:grid",
            True, C_SUB)
        surf.blit(hints, (self.W - hints.get_width() - 10, 15))

    # ── Canvas rendering ────────────────────────────────────────────────
    def _render_canvas(self, surf):
        rect = self._canvas_rect(); N = self.solver.N

        pygame.draw.rect(surf, (8, 8, 14), rect, border_radius=4)
        pygame.draw.rect(surf, C_SURF0, rect.inflate(2, 2), 1, border_radius=5)

        # Density -> RGB
        d = self.solver.d[1:N+1, 1:N+1].T
        dmax = max(d.max(), 1e-6)
        dn = np.clip(d / dmax, 0, 1) ** 0.6
        rgb = np.ascontiguousarray(apply_colormap(dn, self.cmap_name))
        grid_surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        grid_surf = pygame.transform.smoothscale(grid_surf, (rect.w, rect.h))
        surf.blit(grid_surf, rect.topleft)

        # Overlays
        if self.show_tmp:
            self._blit_overlay(surf, rect, N, self._tmp_rgb, 0.35)
        if self.show_prs:
            self._blit_overlay(surf, rect, N, self._prs_rgb, 0.30)
        if self.show_vor:
            self._blit_overlay(surf, rect, N, self._vor_rgb, 0.35)
        self._draw_obstacles(surf, rect, N)
        if self.show_vel:
            self._draw_arrows(surf, rect, N)

        # Source markers
        for sx, sy, *_ in self.solver.sources:
            mx = rect.x + sx/(N+2)*rect.w
            my = rect.y + sy/(N+2)*rect.h
            pygame.draw.circle(surf, (255, 220, 50), (int(mx), int(my)), 6, 2)
            pygame.draw.line(surf, (255, 220, 50), (int(mx)-4, int(my)), (int(mx)+4, int(my)), 1)
            pygame.draw.line(surf, (255, 220, 50), (int(mx), int(my)-4), (int(mx), int(my)+4), 1)

        # Mode banner
        if self.obstacle_mode:
            txt = font(13, True).render("WALL MODE  (R-click to erase)", True, C_RED)
            surf.blit(txt, (rect.centerx - txt.get_width()//2, rect.y + 8))

        if self.show_bar:
            self._draw_colorbar(surf, rect, dmax)

    def _blit_overlay(self, surf, rect, N, fn, alpha):
        rgb = fn(N)
        o = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        o = pygame.transform.smoothscale(o, (rect.w, rect.h))
        o.set_alpha(int(alpha * 255))
        surf.blit(o, rect.topleft)

    def _tmp_rgb(self, N):
        T = self.solver.T[1:N+1,1:N+1].T; tm = max(T.max(), 1e-6); tn = np.clip(T/tm, 0, 1)
        return np.stack([(tn*220+20).astype(np.uint8), (tn*120+10).astype(np.uint8),
                         np.full(N*N, 15, dtype=np.uint8).reshape(N, N)], axis=-1)

    def _prs_rgb(self, N):
        pf = self.solver.debug.get('pressure')
        if pf is None: return np.zeros((N,N,3), dtype=np.uint8)
        d = pf[1:N+1,1:N+1].T; mx = max(np.abs(d).max(), 1e-8); dn = d/mx
        pos = np.clip(dn,0,1); neg = np.clip(-dn,0,1)
        return np.stack([(pos*200+30).astype(np.uint8), np.full(N*N,30,dtype=np.uint8).reshape(N,N),
                         (neg*200+30).astype(np.uint8)], axis=-1)

    def _vor_rgb(self, N):
        wf = self.solver.debug.get('vorticity')
        if wf is None: return np.zeros((N,N,3), dtype=np.uint8)
        w = wf[1:N+1,1:N+1].T; wn = np.clip(w/max(np.abs(w).max(),1e-8),-1,1)
        pos = np.clip(wn,0,1); neg = np.clip(-wn,0,1)
        return np.stack([(pos*220+20).astype(np.uint8), np.full(N*N,20,dtype=np.uint8).reshape(N,N),
                         (neg*220+20).astype(np.uint8)], axis=-1)

    def _draw_obstacles(self, surf, rect, N):
        obs = self.solver.obstacle[1:N+1,1:N+1].T
        if not obs.any(): return
        rgba = np.zeros((N,N,4), dtype=np.uint8)
        rgba[obs,0]=55; rgba[obs,1]=60; rgba[obs,2]=75; rgba[obs,3]=220
        try:
            from scipy.ndimage import binary_dilation
            edge = binary_dilation(obs, iterations=1) & ~obs
            if edge.any():
                rgba[edge,0]=100; rgba[edge,1]=110; rgba[edge,2]=140; rgba[edge,3]=180
        except ImportError: pass
        pix = np.zeros((N,N,4), dtype=np.uint8)
        pix[:,:,0:3] = np.transpose(rgba[:,:,0:3], (1,0,2))
        pix[:,:,3] = rgba[:,:,3].T
        s = pygame.image.frombuffer(pix.tobytes(), (N,N), "RGBA").convert_alpha()
        surf.blit(pygame.transform.smoothscale(s, (rect.w, rect.h)), rect.topleft)

    def _draw_arrows(self, surf, rect, N):
        step = max(N//22, 1); vmax = max(self.solver.max_velocity(), 0.1)
        sc = min(rect.w, rect.h)/(N+2)*0.8/vmax*step; sc = max(0.5, min(sc, 30))
        for i in range(1, N+1, step):
            for j in range(1, N+1, step):
                if self.solver.obstacle[i,j]: continue
                ux, vy = self.solver.u[i,j], self.solver.v[i,j]
                if abs(ux)+abs(vy) < vmax*0.01: continue
                sx = rect.x + i/(N+2)*rect.w; sy = rect.y + j/(N+2)*rect.h
                pygame.draw.line(surf, (130,200,255), (int(sx),int(sy)),
                                 (int(sx+ux*sc), int(sy+vy*sc)), 1)

    def _draw_colorbar(self, surf, rect, dmax):
        cw, ch = 16, 100; x0 = rect.right-cw-12; y0 = rect.bottom-ch-24
        bg = pygame.Surface((cw+48, ch+24), pygame.SRCALPHA); bg.fill((20,20,35,160))
        surf.blit(bg, (x0-6, y0-14))
        for i in range(ch):
            t = 1.0-i/ch; rgb = apply_colormap(np.array([[t]]), self.cmap_name)[0,0]
            pygame.draw.line(surf, (int(rgb[0]),int(rgb[1]),int(rgb[2])),
                             (x0,y0+i),(x0+cw,y0+i))
        pygame.draw.rect(surf, C_SUB, (x0-1,y0-1,cw+2,ch+2), 1)
        mf = mono(9)
        surf.blit(mf.render(f"{dmax:.0f}", True, C_SUB), (x0+cw+3, y0))
        surf.blit(mf.render("0", True, C_SUB), (x0+cw+3, y0+ch-9))

    def _draw_status(self, surf):
        y = self.H - self.STATUS_H
        pygame.draw.rect(surf, C_MANTLE, (0, y, self.W, self.STATUS_H))
        pygame.draw.line(surf, C_SURF0, (0, y), (self.W, y))
        fps = len(self._frame_times)
        s = self.solver
        txt = mono(11).render(
            f"FPS:{fps}  Solve:{s.solve_ms:.1f}ms  Grid:{s.N}^2  "
            f"Step:{s.step_count}  MaxV:{s.max_velocity():.1f}  "
            f"SumD:{s.total_density():.0f}  Src:{len(s.sources)}  "
            f"Buoy:{s.buoyancy:.1f}  Vort:{s.vort_conf:.1f}", True, C_SUB)
        surf.blit(txt, (10, y+5))

    # ── Events ──────────────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)

            btns = self._build_toolbar_buttons()

            if ev.type == pygame.MOUSEBUTTONDOWN:
                # Toolbar
                if ev.button == 1 and ev.pos[1] < self.TOOLBAR_H:
                    self._handle_toolbar_click(ev.pos, btns)
                    continue

                rect = self._canvas_rect()
                if not rect.collidepoint(ev.pos):
                    continue

                self.mouse_down = ev.button
                self.last_mouse = ev.pos
                gi, gj = self._screen_to_grid(ev.pos)

                if self.obstacle_mode:
                    if ev.button == 1:
                        self.solver.set_obstacle(gi, gj, r=2, value=True)
                    elif ev.button == 3:
                        self.solver.set_obstacle(gi, gj, r=3, value=False)
                elif ev.button == 2:  # middle click = place emitter
                    self.solver.sources.append((gi, gj, 18, 0, -5, 14))
                elif ev.button == 1:
                    self.solver.add_density(gi, gj, 80, r=4)
                    self.solver.add_velocity(gi, gj, 0, -6, r=4)
                    self.solver.add_temperature(gi, gj, 15, r=4)

            elif ev.type == pygame.MOUSEMOTION and self.mouse_down is not None:
                rect = self._canvas_rect()
                if rect.collidepoint(ev.pos):
                    gi, gj = self._screen_to_grid(ev.pos)
                    if self.obstacle_mode:
                        if self.mouse_down == 1:
                            self.solver.set_obstacle(gi, gj, r=2, value=True)
                        elif self.mouse_down == 3:
                            self.solver.set_obstacle(gi, gj, r=3, value=False)
                    elif self.last_mouse:
                        dx = ev.pos[0]-self.last_mouse[0]
                        dy = ev.pos[1]-self.last_mouse[1]
                        if self.mouse_down == 1:
                            self.solver.add_density(gi, gj, 40, r=3)
                            self.solver.add_velocity(gi, gj, dx*0.5, dy*0.5, r=3)
                            self.solver.add_temperature(gi, gj, 8, r=3)
                        elif self.mouse_down == 3:
                            self.solver.add_velocity(gi, gj, dx*1.2, dy*1.2, r=5)
                self.last_mouse = ev.pos

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == self.mouse_down:
                    self.mouse_down = None; self.last_mouse = None

            elif ev.type == pygame.MOUSEWHEEL:
                # Scroll up/down adjusts buoyancy
                self.solver.buoyancy = np.clip(self.solver.buoyancy + ev.y * 0.2, 0, 20)

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    self.solver.reset()
                elif ev.key == pygame.K_s:
                    self.solver.step()
                elif ev.key == pygame.K_v:
                    self.show_vel = not self.show_vel
                elif ev.key == pygame.K_w:
                    self.show_vor = not self.show_vor
                elif ev.key == pygame.K_p:
                    self.show_prs = not self.show_prs
                elif ev.key == pygame.K_t:
                    self.show_tmp = not self.show_tmp
                elif ev.key == pygame.K_b:
                    self.show_bar = not self.show_bar
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6):
                    self.cmap_idx = ev.key - pygame.K_1
                elif ev.key == pygame.K_RIGHTBRACKET:
                    sizes = [64,80,100,128,160,200]
                    cur = self.solver.N
                    try:
                        idx = min(sizes.index(cur)+1, len(sizes)-1)
                    except ValueError:
                        idx = 3
                    self.solver.resize(sizes[idx])
                elif ev.key == pygame.K_LEFTBRACKET:
                    sizes = [64,80,100,128,160,200]
                    cur = self.solver.N
                    try:
                        idx = max(sizes.index(cur)-1, 0)
                    except ValueError:
                        idx = 3
                    self.solver.resize(sizes[idx])
                elif ev.key == pygame.K_c:
                    # Clear all sources
                    self.solver.sources.clear()
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
            btns = self._build_toolbar_buttons()
            self._draw_toolbar(self.screen, btns)
            self._render_canvas(self.screen)
            self._draw_status(self.screen)
            pygame.display.flip()

            now = time.perf_counter()
            self._frame_times.append(now)
            self._frame_times = [t for t in self._frame_times if now - t < 1.0]
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Playground().run()