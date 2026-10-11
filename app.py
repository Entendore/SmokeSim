#!/usr/bin/env python3
"""
Educational Smoke Simulation — Clean Reference Version
=======================================================
Inspired by Sebastian Lague's "Coding Adventure: Fluid Simulation"
https://www.youtube.com/watch?v=Q78wvrQ9xsU

Install:  pip install PySide6 numpy scipy
"""

import sys, time
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QTextBrowser, QGroupBox, QFrame, QSplitter, QSizePolicy,
    QScrollArea, QGridLayout, QSpinBox, QTabWidget, QToolBar
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRect, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QFont, QCursor, QAction

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
#  SIMULATION CANVAS
# ══════════════════════════════════════════════════════════════════════════════
class SmokeCanvas(QWidget):
    def __init__(self, solver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self.cmap_name = "Fire"
        self.show_velocity = self.show_vorticity = self.show_pressure = self.show_temperature = False
        self.show_colorbar = True; self.source_mode = False; self.obstacle_mode = False
        self._last = None; self._btn = None
        self._pix_density = None; self._pix_overlay = None
        self.setMinimumSize(380,380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True); self.setFocusPolicy(Qt.StrongFocus)

    def _rect(self):
        s = min(self.width(), self.height()) - 20
        return QRect((self.width()-s)//2, (self.height()-s)//2, s, s)

    def _s2g(self, pos):
        N = self.solver.N; r = self._rect()
        return (int(np.clip((pos.x()-r.x())/r.width()*(N+2),1,N)),
                int(np.clip((pos.y()-r.y())/r.height()*(N+2),1,N)))

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(14,14,24))
        N = self.solver.N; rect = self._rect()
        p.setPen(QPen(QColor(69,71,90),2)); p.setBrush(QColor(8,8,14))
        p.drawRoundedRect(rect.adjusted(-2,-2,2,2),4,4)
        d = self.solver.d[1:N+1,1:N+1].T; dmax = max(d.max(), 1e-6)
        dn = np.clip(d/dmax,0,1)**0.6
        rgb = np.ascontiguousarray(apply_colormap(dn, self.cmap_name))
        h,w,_ = rgb.shape; img = QImage(rgb.data,w,h,3*w,QImage.Format_RGB888)
        self._pix_density = QPixmap.fromImage(img.copy())
        p.drawPixmap(rect, self._pix_density)
        if self.show_temperature: self._draw_temperature(p,rect)
        if self.show_pressure: self._draw_pressure(p,rect)
        if self.show_vorticity: self._draw_vorticity(p,rect)
        self._draw_obstacles(p,rect)
        if self.show_velocity: self._draw_velocity(p,rect)
        for sx,sy,*_ in self.solver.sources:
            mx=rect.x()+sx/(N+2)*rect.width(); my=rect.y()+sy/(N+2)*rect.height()
            p.setPen(QPen(QColor(255,220,50,200),2)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(mx,my),7,7); p.drawLine(QPointF(mx-4,my),QPointF(mx+4,my)); p.drawLine(QPointF(mx,my-4),QPointF(mx,my+4))
        if self.show_colorbar: self._draw_colorbar(p,dmax)
        if self.source_mode or self.obstacle_mode:
            txt="📍 Source Mode" if self.source_mode else "🧱 Wall Mode (Right-click: Erase)"
            p.setPen(QPen(QColor(255,220,50,220) if self.source_mode else QColor(255,100,100,220),1))
            p.setFont(QFont("Segoe UI",11,QFont.Bold)); p.drawText(rect,Qt.AlignTop|Qt.AlignHCenter,txt)
        p.end()

    def _make_overlay_pix(self, rgb_array):
        rgb=np.ascontiguousarray(rgb_array); h,w,_=rgb.shape
        img=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888); return QPixmap.fromImage(img.copy())

    def _draw_temperature(self,p,rect):
        N=self.solver.N; T=self.solver.T[1:N+1,1:N+1].T; tm=max(T.max(),1e-6); tn=np.clip(T/tm,0,1)
        r=(tn*220+20).astype(np.uint8); g=(tn*120+10).astype(np.uint8); b=np.full_like(r,15,dtype=np.uint8)
        self._pix_overlay=self._make_overlay_pix(np.stack([r,g,b],axis=-1))
        p.setOpacity(0.35); p.drawPixmap(rect,self._pix_overlay); p.setOpacity(1)

    def _draw_pressure(self,p,rect):
        pf=self.solver.debug.get('pressure')
        if pf is None: return
        N=self.solver.N; d=pf[1:N+1,1:N+1].T; mx=max(np.abs(d).max(),1e-8); dn=d/mx
        pos=np.clip(dn,0,1); neg=np.clip(-dn,0,1)
        r=(pos*200+30).astype(np.uint8); g=np.full_like(r,30,dtype=np.uint8); b=(neg*200+30).astype(np.uint8)
        self._pix_overlay=self._make_overlay_pix(np.stack([r,g,b],axis=-1))
        p.setOpacity(0.3); p.drawPixmap(rect,self._pix_overlay); p.setOpacity(1)

    def _draw_vorticity(self,p,rect):
        wf=self.solver.debug.get('vorticity')
        if wf is None: return
        N=self.solver.N; w=wf[1:N+1,1:N+1].T; wn=np.clip(w/max(np.abs(w).max(),1e-8),-1,1)
        pos=np.clip(wn,0,1); neg=np.clip(-wn,0,1)
        r=(pos*220+20).astype(np.uint8); g=np.full_like(r,20,dtype=np.uint8); b=(neg*220+20).astype(np.uint8)
        self._pix_overlay=self._make_overlay_pix(np.stack([r,g,b],axis=-1))
        p.setOpacity(0.35); p.drawPixmap(rect,self._pix_overlay); p.setOpacity(1)

    def _draw_obstacles(self,p,rect):
        N=self.solver.N; obs=self.solver.obstacle[1:N+1,1:N+1].T
        if not obs.any(): return
        r=np.where(obs,55,0).astype(np.uint8); g=np.where(obs,60,0).astype(np.uint8)
        b=np.where(obs,75,0).astype(np.uint8); a=np.where(obs,220,0).astype(np.uint8)
        rgba=np.ascontiguousarray(np.stack([r,g,b,a],axis=-1))
        h,w,_=rgba.shape; img=QImage(rgba.data,w,h,4*w,QImage.Format_RGBA8888)
        pix=QPixmap.fromImage(img.copy()); p.drawPixmap(rect,pix)
        try:
            from scipy.ndimage import binary_dilation
            edge=binary_dilation(obs,iterations=1)&~obs
            if edge.any():
                er=np.where(edge,100,0).astype(np.uint8); eg=np.where(edge,110,0).astype(np.uint8)
                eb=np.where(edge,140,0).astype(np.uint8); ea=np.where(edge,180,0).astype(np.uint8)
                ergba=np.ascontiguousarray(np.stack([er,eg,eb,ea],axis=-1))
                eimg=QImage(ergba.data,w,h,4*w,QImage.Format_RGBA8888)
                epix=QPixmap.fromImage(eimg.copy()); p.drawPixmap(rect,epix)
        except ImportError: pass

    def _draw_velocity(self,p,rect):
        N=self.solver.N; step=max(N//22,1)
        vmax=max(self.solver.max_velocity(),0.1)
        sc=min(rect.width(),rect.height())/(N+2)*0.8/vmax*step; sc=max(0.5,min(sc,30))
        p.setPen(QPen(QColor(130,200,255,110),1))
        for i in range(1,N+1,step):
            for j in range(1,N+1,step):
                if self.solver.obstacle[i,j]: continue
                ux=self.solver.u[i,j]; vy=self.solver.v[i,j]
                if abs(ux)+abs(vy)<vmax*0.01: continue
                sx=rect.x()+i/(N+2)*rect.width(); sy=rect.y()+j/(N+2)*rect.height()
                p.drawLine(QPointF(sx,sy),QPointF(sx+ux*sc,sy+vy*sc))

    def _draw_colorbar(self,p,dmax):
        cw,ch=18,120; margin=10
        x0=self.width()-cw-margin-5; y0=self.height()-ch-margin-25
        p.setPen(Qt.NoPen); p.setBrush(QColor(20,20,35,180)); p.drawRoundedRect(x0-5,y0-15,cw+45,ch+30,4,4)
        p.setPen(QColor(200,200,200)); p.setFont(QFont("Consolas",8)); p.drawText(x0-2,y0-3,"ρ")
        for i in range(ch):
            t=1.0-i/ch; rgb=apply_colormap(np.array([[t]]),self.cmap_name)[0,0]
            p.setPen(QColor(int(rgb[0]),int(rgb[1]),int(rgb[2]))); p.drawLine(x0,y0+i,x0+cw,y0+i)
        p.setPen(QColor(200,200,200)); p.drawRect(x0-1,y0-1,cw+2,ch+2)
        p.drawText(x0+cw+4,y0+8,f"{dmax:.0f}"); p.drawText(x0+cw+4,y0+ch,"0")

    def mousePressEvent(self,ev):
        self._btn=ev.button(); self._last=ev.position(); gx,gy=self._s2g(ev.position())
        if self.obstacle_mode:
            self.solver.set_obstacle(gx,gy,r=2,value=(ev.button()==Qt.LeftButton))
            if ev.button()==Qt.RightButton: self.solver.set_obstacle(gx,gy,r=3,value=False)
            return
        if self.source_mode and ev.button()==Qt.LeftButton:
            self.solver.sources.append((gx,gy,15,0,-4,12)); self.source_mode=False; return
        if ev.button()==Qt.LeftButton:
            self.solver.add_density(gx,gy,80,r=4); self.solver.add_velocity(gx,gy,0,-6,r=4); self.solver.add_temperature(gx,gy,15,r=4)

    def mouseMoveEvent(self,ev):
        if self._btn is None: return
        gx,gy=self._s2g(ev.position())
        if self.obstacle_mode:
            self.solver.set_obstacle(gx,gy,r=2,value=(self._btn==Qt.LeftButton))
            if self._btn==Qt.RightButton: self.solver.set_obstacle(gx,gy,r=3,value=False)
            self._last=ev.position(); return
        if self._last:
            dx=ev.position().x()-self._last.x(); dy=ev.position().y()-self._last.y()
            if self._btn==Qt.LeftButton:
                self.solver.add_density(gx,gy,40,r=3); self.solver.add_velocity(gx,gy,dx*.5,dy*.5,r=3); self.solver.add_temperature(gx,gy,8,r=3)
            elif self._btn==Qt.RightButton: self.solver.add_velocity(gx,gy,dx*1.2,dy*1.2,r=5)
        self._last=ev.position()

    def mouseReleaseEvent(self,_): self._btn=None; self._last=None


# ══════════════════════════════════════════════════════════════════════════════
#  LESSON DATA — SIMPLE REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
YT_URL = "https://www.youtube.com/watch?v=Q78wvrQ9xsU"
REF_HTML = f"<hr style='border:1px solid #313244; margin-top:12px;'><p style='font-size:11px; color:#a6adc8;'>📚 <b>Reference:</b> <a href='{YT_URL}' style='color:#89b4fa;'>Sebastian Lague — Coding Adventure: Fluid Simulation</a></p>"

LESSONS = [
    { "title": "1. What is Fluid Sim?", "preset": "candle", "highlights": ["buoyancy"],
      "html": """<h3 style='color:#fab387;'>🧪 What is Fluid Simulation?</h3><p>Solving the <b>Navier-Stokes equations</b> — governing weather, oceans, and smoke.</p><p><b>Why is it hard?</b> Non-linear + incompressibility constraint.</p><p><b>Stam's Insight:</b> Trade accuracy for guaranteed stability.</p><h4 style='color:#89b4fa;'>🎮 Try It</h4><p>Click <b>"Run This Demo"</b> for a rising plume. Drag mouse to interact!</p>""" },
    { "title": "2. Navier-Stokes Eqns", "preset": "default", "highlights": ["viscosity"],
      "html": """<h3 style='color:#fab387;'>📐 Navier-Stokes Equations</h3><p style='font-size:14px;text-align:center;color:#a6e3a1;'>∂<b>u</b>/∂t = −(<b>u</b>·∇)<b>u</b> + ν∇²<b>u</b> − ∇p + <b>f</b></p><table style='color:#cdd6f4;font-size:12px;margin:6px;'><tr><td style='padding:3px;color:#f38ba8;'>−(<b>u</b>·∇)<b>u</b></td><td>Advection</td></tr><tr><td style='padding:3px;color:#f9e2af;'>ν∇²<b>u</b></td><td>Viscous Diffusion</td></tr><tr><td style='padding:3px;color:#89b4fa;'>−∇p</td><td>Pressure</td></tr><tr><td style='padding:3px;color:#a6e3a1;'><b>f</b></td><td>External Forces</td></tr></table><p><b>∇·<b>u</b> = 0</b> (Incompressibility)</p>""" },
    { "title": "3. Advection", "preset": "advection", "highlights": ["dt"],
      "html": """<h3 style='color:#fab387;'>🌊 Advection</h3><p>Carrying quantities along the velocity field.</p><p><b>Naïve:</b> <code>x_new = x + dt·u(x)</code> ❌ Unstable!</p><p><b>Semi-Lagrangian:</b> Trace backwards. Sample via interpolation. ✅ Stable!</p>""" },
    { "title": "4. Diffusion", "preset": "diffusion", "highlights": ["viscosity","diffusion"],
      "html": """<h3 style='color:#fab387;'>🌫️ Diffusion</h3><p>Velocity spreads via <b>viscosity (ν)</b>; density via <b>diffusion (κ)</b>.</p><p><b>Implicit Solve:</b> Gauss-Seidel relaxation. ✅ Unconditionally stable.</p>""" },
    { "title": "5. Pressure", "preset": "pressure", "highlights": [],
      "html": """<h3 style='color:#fab387;'>⚡ Pressure</h3><p>Fixing divergence via <b>Helmholtz-Hodge decomposition</b>.</p><ol><li>Compute ∇·<b>u</b></li><li>Solve Poisson: ∇²p = ∇·<b>u</b></li><li>Subtract ∇p</li></ol><p>Enable <b>Pressure Overlay</b> to visualize!</p>""" },
    { "title": "6. Forces", "preset": "vorticity", "highlights": ["buoyancy","vorticity"],
      "html": """<h3 style='color:#fab387;'>🔥 Forces</h3><p><b>Buoyancy:</b> Temperature drives upward force.</p><p><b>Vorticity Confinement:</b> Re-injects lost swirl energy.</p><p>Try setting vorticity to 0 vs 20!</p>""" },
    { "title": "7. Obstacles", "preset": "obstacle", "highlights": [],
      "html": """<h3 style='color:#fab387;'>🧱 Obstacles</h3><p>Walls enforce zero velocity and alter pressure.</p><p>Toggle <b>Wall Mode</b> in toolbar: Left-draw, Right-erase.</p>""" },
    { "title": "8. 🎮 Playground", "preset": "playground", "highlights": [],
      "html": """<h3 style='color:#fab387;'>🎮 Playground</h3><p>Experiment! Use toolbar buttons to switch modes.</p><ul><li><b>Left-drag:</b> Smoke + Heat + Push</li><li><b>Right-drag:</b> Wind only</li><li><b>Keyboard:</b> Space(Pause), R(Reset), V/W/P/T(Overlays)</li></ul>""" },
]


# ══════════════════════════════════════════════════════════════════════════════
#  EDUCATION PANEL
# ══════════════════════════════════════════════════════════════════════════════
class EducationPanel(QWidget):
    preset_requested = Signal(str)
    lesson_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_lesson = 0; self._build()

    def _build(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        
        # ── TAB 1: LEARN ────────────────────────────────────────────────
        learn_w = QWidget(); learn_l = QVBoxLayout(learn_w); learn_l.setContentsMargins(6,6,6,6)
        
        # Navigation
        nav = QHBoxLayout(); nav.setSpacing(4)
        self.btn_prev = QPushButton("◀"); self.btn_prev.setFixedWidth(32); self.btn_prev.clicked.connect(self._prev_lesson); nav.addWidget(self.btn_prev)
        self.lesson_combo = QComboBox()
        for l in LESSONS: self.lesson_combo.addItem(l["title"])
        self.lesson_combo.currentIndexChanged.connect(self._set_lesson); nav.addWidget(self.lesson_combo,1)
        self.btn_next = QPushButton("▶"); self.btn_next.setFixedWidth(32); self.btn_next.clicked.connect(self._next_lesson); nav.addWidget(self.btn_next)
        learn_l.addLayout(nav)
        
        self.lesson_text = QTextBrowser(); self.lesson_text.setOpenExternalLinks(True); learn_l.addWidget(self.lesson_text,1)
        
        row = QHBoxLayout()
        self.btn_try = QPushButton("🎮  Run This Demo"); self.btn_try.setFixedHeight(38); self.btn_try.setObjectName("tryBtn"); self.btn_try.clicked.connect(self._try_demo); row.addWidget(self.btn_try)
        learn_l.addLayout(row)
        
        self.tabs.addTab(learn_w, "🎓 Learn")
        
        # ── TAB 2: PHYSICS ──────────────────────────────────────────────
        phys_w = QWidget(); phys_l = QVBoxLayout(phys_w); phys_l.setContentsMargins(6,6,6,6)
        phys_inner = QGroupBox("⚛  Parameters"); gl = QGridLayout(phys_inner); gl.setSpacing(6)
        self.sliders = {}
        params = [("viscosity","Viscosity (ν)",0,50,1,1),("diffusion","Diffusion (κ)",0,50,1,1),("buoyancy","Buoyancy (α)",0,100,10,10),("vorticity","Vorticity (ε)",0,250,5,50),("dt","Time Step",1,30,1,10),("decay","Decay (‰)",980,1000,1,997)]
        for ri,(key,label,lo,hi,step,val) in enumerate(params):
            lbl=QLabel(label); sl=QSlider(Qt.Horizontal); sl.setRange(lo,hi); sl.setSingleStep(step); sl.setValue(val)
            vlbl=QLabel(str(val)); vlbl.setMinimumWidth(30); vlbl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            sl.valueChanged.connect(lambda v,l=vlbl:l.setText(str(v)))
            gl.addWidget(lbl,ri,0); gl.addWidget(sl,ri,1); gl.addWidget(vlbl,ri,2)
            self.sliders[key]=(sl,vlbl,lbl)
        phys_l.addWidget(phys_inner); phys_l.addStretch()
        self.tabs.addTab(phys_w, "⚛ Physics")
        
        # ── TAB 3: VISUALS ──────────────────────────────────────────────
        viz_w = QWidget(); viz_l = QVBoxLayout(viz_w); viz_l.setContentsMargins(6,6,6,6)
        viz_inner = QGroupBox("🎨  Overlays & Display"); vl = QVBoxLayout(viz_inner); vl.setSpacing(6)
        r1=QHBoxLayout(); r1.addWidget(QLabel("Colour:")); self.cmap_combo=QComboBox(); self.cmap_combo.addItems(COLORMAPS.keys()); self.cmap_combo.setCurrentText("Fire"); r1.addWidget(self.cmap_combo,1); vl.addLayout(r1)
        r2=QHBoxLayout(); r2.setSpacing(10); self.cb_vel=QCheckBox("Velocity"); self.cb_vor=QCheckBox("Vorticity"); self.cb_prs=QCheckBox("Pressure"); self.cb_tmp=QCheckBox("Temp")
        r2.addWidget(self.cb_vel); r2.addWidget(self.cb_vor); r2.addWidget(self.cb_prs); r2.addWidget(self.cb_tmp); vl.addLayout(r2)
        r3=QHBoxLayout(); r3.addWidget(QLabel("Grid:")); self.grid_combo=QComboBox(); self.grid_combo.addItems(["64","100","128","160"]); self.grid_combo.setCurrentText("100"); r3.addWidget(self.grid_combo,1)
        r3.addWidget(QLabel("Steps/frame:")); self.steps_spin=QSpinBox(); self.steps_spin.setRange(1,8); self.steps_spin.setValue(1); r3.addWidget(self.steps_spin); vl.addLayout(r3)
        self.cb_bar=QCheckBox("Colour bar"); self.cb_bar.setChecked(True); vl.addWidget(self.cb_bar)
        viz_l.addWidget(viz_inner); viz_l.addStretch()
        self.tabs.addTab(viz_w, "🎨 Visuals")
        
        layout.addWidget(self.tabs); self._set_lesson(0)

    def _set_lesson(self,idx):
        if idx<0 or idx>=len(LESSONS): return
        self.current_lesson=idx; lesson=LESSONS[idx]
        self.lesson_text.setHtml(f"<body style='background:#181825;color:#cdd6f4;font-family:Segoe UI;font-size:13px;padding:4px;'>{lesson['html']}{REF_HTML}</body>")
        self.lesson_combo.blockSignals(True); self.lesson_combo.setCurrentIndex(idx); self.lesson_combo.blockSignals(False)
        self.lesson_changed.emit(idx)

    def _prev_lesson(self): self._set_lesson(max(0,self.current_lesson-1))
    def _next_lesson(self): self._set_lesson(min(len(LESSONS)-1,self.current_lesson+1))
    def _try_demo(self): self.preset_requested.emit(LESSONS[self.current_lesson]["preset"])

    def get_params(self):
        s=self.sliders
        return dict(viscosity=s["viscosity"][0].value()*1e-4,diffusion=s["diffusion"][0].value()*1e-4,buoyancy=s["buoyancy"][0].value()/10.0,vorticity=s["vorticity"][0].value()/10.0,dt=s["dt"][0].value()/100.0,decay=s["decay"][0].value()/1000.0)

    def set_params(self,visc,diff,buoy,vort,dt,decay):
        s=self.sliders
        s["viscosity"][0].setValue(int(visc/1e-4)); s["diffusion"][0].setValue(int(diff/1e-4))
        s["buoyancy"][0].setValue(int(buoy*10)); s["vorticity"][0].setValue(int(vort*10))
        s["dt"][0].setValue(int(dt*100)); s["decay"][0].setValue(int(decay*1000))

    def highlight_sliders(self,keys):
        for key,(sl,vlbl,lbl) in self.sliders.items():
            if key in keys:
                sl.setStyleSheet("QSlider::handle:horizontal{background:#f9e2af;}"); vlbl.setStyleSheet("color:#f9e2af;font-weight:bold;"); lbl.setStyleSheet("color:#f9e2af;font-weight:bold;")
            else: sl.setStyleSheet(""); vlbl.setStyleSheet(""); lbl.setStyleSheet("")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧪 Educational Smoke Simulation — Fluid Dynamics")
        self.resize(1320,820)
        self.solver=FluidSolver(N=100); self.canvas=SmokeCanvas(self.solver)
        self.paused=False; self._frame_times=[]
        self._build_ui(); self._apply_style(); self._connect_signals()
        self._apply_preset("candle"); self._start_timer()

    def _build_ui(self):
        tb=QToolBar("Main Controls"); tb.setMovable(False); tb.setObjectName("mainToolBar"); self.addToolBar(tb)
        self.act_pause=QAction("⏸ Pause",self); self.act_pause.setCheckable(True); self.act_pause.setShortcut("Space")
        self.act_step=QAction("⏭ Step",self); self.act_step.setShortcut("S")
        self.act_reset=QAction("🔄 Reset",self); self.act_reset.setShortcut("R")
        tb.addAction(self.act_pause); tb.addAction(self.act_step); tb.addAction(self.act_reset); tb.addSeparator()
        self.act_source=QAction("📍 Source",self); self.act_source.setCheckable(True); self.act_source.setToolTip("Click canvas to place emitter")
        self.act_wall=QAction("🧱 Wall",self); self.act_wall.setCheckable(True); self.act_wall.setToolTip("Left-draw wall, Right-erase")
        tb.addAction(self.act_source); tb.addAction(self.act_wall); tb.addSeparator()
        self.lbl_lesson=QLabel("Lesson: 1"); self.lbl_lesson.setObjectName("toolbarLessonLabel"); tb.addWidget(self.lbl_lesson)
        splitter=QSplitter(Qt.Horizontal); self.setCentralWidget(splitter)
        cf=QFrame(); cl=QVBoxLayout(cf); cl.setContentsMargins(4,4,4,4); cl.addWidget(self.canvas); splitter.addWidget(cf)
        self.edu=EducationPanel(); self.edu.setMinimumWidth(320); self.edu.setMaximumWidth(420); splitter.addWidget(self.edu)
        splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,0)
        self.status=self.statusBar(); self.status.setFont(QFont("Consolas",9))

    def _connect_signals(self):
        e=self.edu; e.preset_requested.connect(self._apply_preset); e.lesson_changed.connect(self._on_lesson_changed)
        self.act_pause.toggled.connect(self._toggle_pause); self.act_step.triggered.connect(self._single_step); self.act_reset.triggered.connect(self._reset)
        self.act_source.toggled.connect(lambda v:setattr(self.canvas,'source_mode',v)); self.act_wall.toggled.connect(self._toggle_wall_mode)
        e.cmap_combo.currentTextChanged.connect(lambda v:setattr(self.canvas,'cmap_name',v))
        e.cb_vel.toggled.connect(lambda v:setattr(self.canvas,'show_velocity',v)); e.cb_vor.toggled.connect(lambda v:setattr(self.canvas,'show_vorticity',v))
        e.cb_prs.toggled.connect(lambda v:setattr(self.canvas,'show_pressure',v)); e.cb_tmp.toggled.connect(lambda v:setattr(self.canvas,'show_temperature',v))
        e.cb_bar.toggled.connect(lambda v:setattr(self.canvas,'show_colorbar',v)); e.grid_combo.currentTextChanged.connect(self._set_grid)
        for key,(sl,_,_) in e.sliders.items(): sl.valueChanged.connect(self._sync_params)

    def _on_lesson_changed(self,idx):
        lesson=LESSONS[idx]; self.lbl_lesson.setText(f"Lesson: {lesson['title']}"); self.edu.highlight_sliders(lesson.get("highlights",[]))

    def _apply_preset(self,name):
        self.solver.reset(); N=self.solver.N; cx=N//2; cy=N//2
        presets={
            "candle":dict(buoyancy=2.0,vort_conf=3.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-6,12)],cmap="Fire"),
            "default":dict(buoyancy=1.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[],cmap="Fire"),
            "advection":dict(buoyancy=1.0,vort_conf=5.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,sources=[(cx,N-5,20,0,-5,10)],cmap="Cool"),
            "diffusion":dict(buoyancy=1.5,vort_conf=0.0,visc=0.005,diff=0.005,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-5,12)],cmap="Grayscale"),
            "pressure":dict(buoyancy=2.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,25,0,-6,12)],cmap="Fire",overlays={"pressure":True}),
            "vorticity":dict(buoyancy=2.0,vort_conf=20.0,visc=0.0,diff=0.0,decay=0.998,dt=0.1,sources=[(cx,N-5,25,0,-5,12)],cmap="Neon",overlays={"vorticity":True}),
            "obstacle":dict(buoyancy=2.0,vort_conf=8.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[(cx,N-5,30,0,-7,15)],cmap="Thermal",obstacles="wall_with_gap"),
            "playground":dict(buoyancy=1.0,vort_conf=5.0,visc=0.0001,diff=0.0001,decay=0.997,dt=0.1,sources=[],cmap="Fire"),
        }
        p=presets.get(name,presets["default"])
        self.solver.buoyancy=p["buoyancy"]; self.solver.vort_conf=p["vort_conf"]; self.solver.visc=p["visc"]; self.solver.diff=p["diff"]; self.solver.decay=p["decay"]; self.solver.dt=p["dt"]
        for s in p.get("sources",[]): self.solver.sources.append(s)
        if p.get("obstacles")=="wall_with_gap":
            for i in range(1,N+1):
                for j in range(N//2-2,N//2+2):
                    if abs(i-cx)>6: self.solver.obstacle[i,j]=True
        for key,val in p.get("overlays",{}).items():
            if key=="pressure": self.canvas.show_pressure=val; self.edu.cb_prs.setChecked(val)
            elif key=="vorticity": self.canvas.show_vorticity=val; self.edu.cb_vor.setChecked(val)
        self.edu.set_params(self.solver.visc,self.solver.diff,self.solver.buoyancy,self.solver.vort_conf,self.solver.dt,self.solver.decay)
        self.edu.cmap_combo.setCurrentText(p.get("cmap","Fire")); self._on_lesson_changed(self.edu.current_lesson); self.canvas.update()

    def _sync_params(self):
        p=self.edu.get_params(); self.solver.visc=p["viscosity"]; self.solver.diff=p["diffusion"]
        self.solver.buoyancy=p["buoyancy"]; self.solver.vort_conf=p["vorticity"]; self.solver.dt=p["dt"]; self.solver.decay=p["decay"]

    def _set_grid(self,text): N=int(text); self.solver.resize(N); self.canvas.update()
    def _toggle_pause(self,v): self.paused=v; self.act_pause.setText("▶ Play" if v else "⏸ Pause")
    def _toggle_wall_mode(self,v): self.canvas.obstacle_mode=v; self.canvas.setCursor(QCursor(Qt.CrossCursor if v else Qt.ArrowCursor))
    def _single_step(self): self.solver.step(); self.canvas.update()
    def _reset(self): self.solver.reset(); self.canvas.update()

    def _start_timer(self): self.timer=QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(16)

    def _tick(self):
        if not self.paused:
            for _ in range(self.edu.steps_spin.value()): self.solver.step()
        self.canvas.update()
        now=time.perf_counter(); self._frame_times.append(now); self._frame_times=[t for t in self._frame_times if now-t<1.0]
        fps=len(self._frame_times); mv=self.solver.max_velocity(); td=self.solver.total_density(); ns=len(self.solver.sources); sm=self.solver.solve_ms; sc=self.solver.step_count
        self.status.showMessage(f"FPS: {fps}  |  Solve: {sm:.1f}ms  |  Grid: {self.solver.N}²  |  Step: {sc}  |  Max|v|: {mv:.1f}  |  Σρ: {td:.0f}")

    def keyPressEvent(self,ev):
        k=ev.key()
        if k==Qt.Key_Space: self.act_pause.toggle()
        elif k==Qt.Key_R: self._reset()
        elif k==Qt.Key_S: self._single_step()
        elif k==Qt.Key_V: self.edu.cb_vel.toggle()
        elif k==Qt.Key_W: self.edu.cb_vor.toggle()
        elif k==Qt.Key_P: self.edu.cb_prs.toggle()
        elif k==Qt.Key_T: self.edu.cb_tmp.toggle()
        elif k==Qt.Key_Left: self.edu._prev_lesson()
        elif k==Qt.Key_Right: self.edu._next_lesson()
        else: super().keyPressEvent(ev)

    def _apply_style(self): self.setStyleSheet(DARK_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  DARK STYLESHEET
# ══════════════════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI','Helvetica Neue',sans-serif; font-size: 13px; }
QToolBar { background: #181825; border-bottom: 1px solid #313244; spacing: 6px; padding: 4px; }
QToolBar QToolButton { background: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 5px 12px; color: #cdd6f4; }
QToolBar QToolButton:hover { background: #45475a; }
QToolBar QToolButton:checked { background: #89b4fa; color: #1e1e2e; }
#toolbarLessonLabel { color: #a6e3a1; font-weight: bold; padding: 0 10px; font-size: 13px; }
QTabWidget::pane { border: 1px solid #313244; background: #1e1e2e; border-radius: 4px; }
QTabBar::tab { background: #181825; color: #a6adc8; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; border: 1px solid #313244; margin-right: 2px; }
QTabBar::tab:selected { background: #1e1e2e; color: #cdd6f4; border-bottom: 2px solid #89b4fa; }
QTabBar::tab:hover { background: #1e1e2e; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding: 14px 8px 8px 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #89b4fa; }
QSlider::groove:horizontal { height: 5px; background: #45475a; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; background: #89b4fa; border-radius: 7px; margin: -5px 0; }
QSlider::handle:horizontal:hover { background: #b4d0fb; }
QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 5px; padding: 5px 10px; color: #cdd6f4; }
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; }
QPushButton#tryBtn { background-color: #1e6640; border-color: #a6e3a1; color: #a6e3a1; font-weight: bold; font-size: 13px; border-radius: 6px; }
QPushButton#tryBtn:hover { background-color: #2a8050; }
QComboBox { background-color: #313244; border: 1px solid #45475a; border-radius: 5px; padding: 4px 8px; color: #cdd6f4; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4; selection-background-color: #89b4fa; }
QCheckBox { spacing: 5px; color: #cdd6f4; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px; border: 2px solid #585b70; background: #313244; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QSpinBox { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 2px 6px; color: #cdd6f4; }
QTextBrowser { background-color: #181825; border: 1px solid #313244; border-radius: 6px; padding: 6px; color: #cdd6f4; }
QStatusBar { background: #181825; color: #a6adc8; font-size: 11px; border-top: 1px solid #313244; }
QLabel { background: transparent; }
QSplitter::handle { background: #313244; width: 2px; }
QFrame { background: transparent; }
QScrollBar:vertical { background: #181825; width: 8px; border: none; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

def main():
    app=QApplication(sys.argv); app.setStyle("Fusion")
    win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()