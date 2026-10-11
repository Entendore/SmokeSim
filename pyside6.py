import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QSlider, QPushButton, QComboBox, 
                               QGroupBox, QFormLayout, QFrame, QCheckBox, QTabWidget, 
                               QStackedWidget, QDockWidget, QToolBar, QStatusBar, 
                               QScrollArea, QFileDialog, QSizePolicy)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QElapsedTimer
from PySide6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent, QAction, QIcon, QKeyEvent
from OpenGL.GL import *

# ---------------------------------------------------------
# 2D FLUID SOLVER SHADERS (GLSL 3.30)
# ---------------------------------------------------------

VERT_SHADER = """
#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    v_uv.y = 1.0 - v_uv.y; 
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

FRAG_SPLAT = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_target;
uniform vec2 u_point;
uniform float u_radius;
uniform vec4 u_color; 
void main() {
    vec4 base = texture(u_target, v_uv);
    float d = distance(v_uv, u_point);
    float splat = exp(-d * d / u_radius);
    FragColor = base + u_color * splat;
}
"""

FRAG_BUOYANCY = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_density;
uniform float u_dt;
uniform float u_buoyancy;
uniform float u_weight;
void main() {
    vec2 vel = texture(u_velocity, v_uv).xy;
    float smoke = texture(u_density, v_uv).x;
    float temp = texture(u_density, v_uv).y;
    float force = (temp * u_buoyancy) - (smoke * u_weight);
    vel.y += force * u_dt;
    FragColor = vec4(vel, 0.0, 1.0);
}
"""

FRAG_ADVECT_SL = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_source;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform vec3 u_dissipation; // x=vel, y=smoke, z=temp
void main() {
    vec2 vel = texture(u_velocity, v_uv).xy;
    vec2 uv_prev = v_uv - vel * u_dt;
    FragColor = texture(u_source, uv_prev) * vec4(u_dissipation, 1.0);
}
"""

FRAG_ADVECT_MACCORMACK = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_source;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform vec3 u_dissipation; // x=vel, y=smoke, z=temp
void main() {
    vec2 vel = texture(u_velocity, v_uv).xy;
    vec2 uv_prev = v_uv - vel * u_dt;
    vec4 phi_n = texture(u_source, uv_prev);
    vec2 vel_prev = texture(u_velocity, uv_prev).xy;
    vec2 uv_next = uv_prev + vel_prev * u_dt;
    vec4 phi_np1_hat = texture(u_source, uv_next);
    vec4 phi_corrected = phi_n + 0.5 * (texture(u_source, v_uv) - phi_np1_hat);
    vec4 min_n = min(min(texture(u_source, uv_prev + vec2(u_texelSize.x, 0.0)), 
                         texture(u_source, uv_prev - vec2(u_texelSize.x, 0.0))),
                     min(texture(u_source, uv_prev + vec2(0.0, u_texelSize.y)), 
                         texture(u_source, uv_prev - vec2(0.0, u_texelSize.y))));
    vec4 max_n = max(max(texture(u_source, uv_prev + vec2(u_texelSize.x, 0.0)), 
                         texture(u_source, uv_prev - vec2(u_texelSize.x, 0.0))),
                     max(texture(u_source, uv_prev + vec2(0.0, u_texelSize.y)), 
                         texture(u_source, uv_prev - vec2(0.0, u_texelSize.y))));
    FragColor = clamp(phi_corrected, min_n, max_n) * vec4(u_dissipation, 1.0);
}
"""

FRAG_DIVERGENCE = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main() {
    float L = texture(u_velocity, v_uv - vec2(u_texelSize.x, 0.0)).x;
    float R = texture(u_velocity, v_uv + vec2(u_texelSize.x, 0.0)).x;
    float B = texture(u_velocity, v_uv - vec2(0.0, u_texelSize.y)).y;
    float T = texture(u_velocity, v_uv + vec2(0.0, u_texelSize.y)).y;
    FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}
"""

# Red-Black Gauss-Seidel Solver
FRAG_JACOBI = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform vec2 u_texelSize;
uniform int u_pass; // 0 for Red, 1 for Black
void main() {
    vec2 coord = floor(v_uv / u_texelSize);
    float checker = mod(coord.x + coord.y, 2.0);
    
    if (int(checker) == u_pass) {
        // Update this cell
        float L = texture(u_pressure, v_uv - vec2(u_texelSize.x, 0.0)).x;
        float R = texture(u_pressure, v_uv + vec2(u_texelSize.x, 0.0)).x;
        float B = texture(u_pressure, v_uv - vec2(0.0, u_texelSize.y)).x;
        float T = texture(u_pressure, v_uv + vec2(0.0, u_texelSize.y)).x;
        float div = texture(u_divergence, v_uv).x;
        FragColor = vec4((L + R + B + T - div) * 0.25, 0.0, 0.0, 1.0);
    } else {
        // Keep old value
        FragColor = texture(u_pressure, v_uv);
    }
}
"""

FRAG_GRADIENT_SUB = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform vec2 u_texelSize;
void main() {
    float L = texture(u_pressure, v_uv - vec2(u_texelSize.x, 0.0)).x;
    float R = texture(u_pressure, v_uv + vec2(u_texelSize.x, 0.0)).x;
    float B = texture(u_pressure, v_uv - vec2(0.0, u_texelSize.y)).x;
    float T = texture(u_pressure, v_uv + vec2(0.0, u_texelSize.y)).x;
    vec2 vel = texture(u_velocity, v_uv).xy;
    vel -= 0.5 * vec2(R - L, T - B);
    FragColor = vec4(vel, 0.0, 1.0);
}
"""

FRAG_BOUNDARY = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main() {
    vec2 vel = texture(u_velocity, v_uv).xy;
    if (v_uv.x < u_texelSize.x) vel.x = abs(vel.x);
    else if (v_uv.x > 1.0 - u_texelSize.x) vel.x = -abs(vel.x);
    if (v_uv.y < u_texelSize.y) vel.y = abs(vel.y);
    else if (v_uv.y > 1.0 - u_texelSize.y) vel.y = -abs(vel.y);
    FragColor = vec4(vel, 0.0, 1.0);
}
"""

FRAG_VORTICITY = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main() {
    float L = texture(u_velocity, v_uv - vec2(u_texelSize.x, 0.0)).y;
    float R = texture(u_velocity, v_uv + vec2(u_texelSize.x, 0.0)).y;
    float B = texture(u_velocity, v_uv - vec2(0.0, u_texelSize.y)).x;
    float T = texture(u_velocity, v_uv + vec2(0.0, u_texelSize.y)).x;
    FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}
"""

FRAG_VORTICITY_FORCE = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_vorticity;
uniform vec2 u_texelSize;
uniform float u_curl;
uniform float u_dt;
uniform float u_decay;
void main() {
    float L = texture(u_vorticity, v_uv - vec2(u_texelSize.x, 0.0)).x;
    float R = texture(u_vorticity, v_uv + vec2(u_texelSize.x, 0.0)).x;
    float B = texture(u_vorticity, v_uv - vec2(0.0, u_texelSize.y)).x;
    float T = texture(u_vorticity, v_uv + vec2(0.0, u_texelSize.y)).x;
    float C = texture(u_vorticity, v_uv).x;
    vec2 grad = vec2(abs(T) - abs(B), abs(L) - abs(R));
    float grad_len = max(length(grad), 1e-5);
    vec2 N = grad / grad_len;
    vec2 force = u_curl * N * sign(C) * exp(-u_decay * u_dt);
    vec2 vel = texture(u_velocity, v_uv).xy;
    vel += force * u_dt;
    FragColor = vec4(vel, 0.0, 1.0);
}
"""

FRAG_WAVELET_TURBULENCE = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_velocity;
uniform sampler2D u_vorticity;
uniform vec2 u_texelSize;
uniform float u_time;
uniform float u_amplitude;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1.0,0.0)), f.x),
               mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), f.x), f.y);
}
vec2 curl_noise(vec2 p) {
    float e = 0.01;
    float n  = noise(p);
    float nx = noise(p + vec2(e, 0.0));
    float ny = noise(p + vec2(0.0, e));
    return vec2(ny - n, -(nx - n)) / e;
}
void main() {
    vec2 vel = texture(u_velocity, v_uv).xy;
    float strain = abs(texture(u_vorticity, v_uv).x);
    vec2 turb = vec2(0.0);
    float scale = 10.0; float weight = 1.0;
    for(int i=0; i<3; i++) {
        turb += curl_noise(v_uv * scale + u_time * 0.5) * weight;
        scale *= 2.0; weight *= 0.5;
    }
    vel += turb * strain * u_amplitude;
    FragColor = vec4(vel, 0.0, 1.0);
}
"""

FRAG_CURL_NOISE = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform float u_time;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1.0,0.0)), f.x),
               mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), f.x), f.y);
}
void main() {
    float scale = 8.0;
    float psi = noise(v_uv * scale + u_time * 0.2);
    float e = 0.01;
    float dx = (noise((v_uv + vec2(e, 0.0)) * scale + u_time * 0.2) - psi) / e;
    float dy = (noise((v_uv + vec2(0.0, e)) * scale + u_time * 0.2) - psi) / e;
    FragColor = vec4(vec2(dy, -dx) * 2.0, 0.0, 1.0);
}
"""

FRAG_PARTICLE_UPDATE = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_particles;
uniform sampler2D u_velocity;
uniform float u_dt;
uniform float u_time;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
    vec4 p = texture(u_particles, v_uv);
    p.w += u_dt * 0.5;
    if (p.w > 5.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) {
        p.x = 0.5 + (hash(v_uv * u_time) - 0.5) * 0.1;
        p.y = 0.05 + (hash(v_uv * u_time * 2.0) - 0.5) * 0.05;
        p.w = 0.0;
    }
    vec2 vel = texture(u_velocity, p.xy).xy;
    p.xy += vel * u_dt * 4.0;
    p.z = length(vel);
    FragColor = p;
}
"""

VERT_PARTICLE = """
#version 330 core
out vec4 v_color;
uniform sampler2D u_particles;
uniform float u_size;
vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0); vec3 c1 = vec3(0.05, 0.03, 0.1); vec3 c2 = vec3(0.3, 0.1, 0.3);
    vec3 c3 = vec3(0.6, 0.2, 0.4); vec3 c4 = vec3(0.9, 0.4, 0.3); vec3 c5 = vec3(1.0, 0.8, 0.5);
    if(t < 0.2) return mix(c0, c1, t/0.2); if(t < 0.4) return mix(c1, c2, (t-0.2)/0.2);
    if(t < 0.6) return mix(c2, c3, (t-0.4)/0.2); if(t < 0.8) return mix(c3, c4, (t-0.6)/0.2);
    if(t < 0.95) return mix(c4, c5, (t-0.8)/0.15); return mix(c5, vec3(1.0), (t-0.95)/0.05);
}
void main() {
    ivec2 texSize = textureSize(u_particles, 0);
    int idx = gl_VertexID;
    ivec2 pix = ivec2(idx % texSize.x, idx / texSize.x);
    vec4 p = texelFetch(u_particles, pix, 0);
    gl_Position = vec4(p.xy * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = u_size;
    float speed_norm = clamp(p.z * 1.5, 0.0, 1.0);
    float alpha = 1.0 - clamp(p.w / 5.0, 0.0, 1.0);
    v_color = vec4(magma(speed_norm), alpha * 0.8);
}
"""

FRAG_PARTICLE = """
#version 330 core
in vec4 v_color;
out vec4 FragColor;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = dot(uv, uv);
    if (d > 1.0) discard;
    float alpha = exp(-d * 4.0) * v_color.a;
    FragColor = vec4(v_color.rgb * alpha, alpha);
}
"""

FRAG_BRIGHT_PASS = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_image;
void main() {
    vec3 col = texture(u_image, v_uv).rgb;
    float brightness = dot(col, vec3(0.2126, 0.7152, 0.0722));
    FragColor = vec4(col * smoothstep(0.4, 1.0, brightness), 1.0);
}
"""

FRAG_BLUR = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_image;
uniform vec2 u_texelSize;
uniform bool u_horizontal;
void main() {
    vec2 tex_offset = 1.0 / u_texelSize;
    vec3 result = texture(u_image, v_uv).rgb * 0.227027;
    if (u_horizontal) {
        result += texture(u_image, v_uv + vec2(tex_offset.x, 0.0)).rgb * 0.1945946;
        result += texture(u_image, v_uv - vec2(tex_offset.x, 0.0)).rgb * 0.1945946;
        result += texture(u_image, v_uv + vec2(tex_offset.x * 2.0, 0.0)).rgb * 0.1216216;
        result += texture(u_image, v_uv - vec2(tex_offset.x * 2.0, 0.0)).rgb * 0.1216216;
    } else {
        result += texture(u_image, v_uv + vec2(0.0, tex_offset.y)).rgb * 0.1945946;
        result += texture(u_image, v_uv - vec2(0.0, tex_offset.y)).rgb * 0.1945946;
        result += texture(u_image, v_uv + vec2(0.0, tex_offset.y * 2.0)).rgb * 0.1216216;
        result += texture(u_image, v_uv - vec2(0.0, tex_offset.y * 2.0)).rgb * 0.1216216;
    }
    FragColor = vec4(result, 1.0);
}
"""

FRAG_DISPLAY_2D = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_density;
uniform sampler2D u_bloom;
uniform sampler2D u_pressure;
uniform sampler2D u_vorticity;
uniform sampler2D u_velocity;
uniform vec3 u_color_smoke;
uniform vec3 u_color_fire;
uniform vec3 u_color_hot;
uniform int u_render_mode;
uniform float u_light_intensity;
uniform float u_light_angle;
uniform float u_detail_strength;
uniform bool u_bicubic;
uniform vec2 u_texelSize;
uniform float u_time;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1.0,0.0)), f.x),
               mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    for(int i=0; i<4; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}

vec4 textureBicubic(sampler2D tex, vec2 uv) {
    vec2 texSize = 1.0 / u_texelSize;
    vec2 pix = uv * texSize - 0.5;
    vec2 f = fract(pix);
    vec2 c = floor(pix) + 0.5;
    vec4 sum = vec4(0.0);
    float offsets[4]; offsets[0] = -1.0; offsets[1] = 0.0; offsets[2] = 1.0; offsets[3] = 2.0;
    for(int y=0; y<4; y++){
        for(int x=0; x<4; x++){
            float wx2 = (x==0) ? -0.5*f.x + f.x*f.x - 0.5*f.x*f.x*f.x :
                       (x==1) ? 1.5 - 2.5*f.x*f.x + 1.5*f.x*f.x*f.x :
                       (x==2) ? -1.5 + 2.0*f.x + 2.0*f.x*f.x - 1.5*f.x*f.x*f.x :
                                0.5 - 0.5*f.x*f.x + 0.5*f.x*f.x*f.x;
            float wy2 = (y==0) ? -0.5*f.y + f.y*f.y - 0.5*f.y*f.y*f.y :
                       (y==1) ? 1.5 - 2.5*f.y*f.y + 1.5*f.y*f.y*f.y :
                       (y==2) ? -1.5 + 2.0*f.y + 2.0*f.y*f.y - 1.5*f.y*f.y*f.y :
                                0.5 - 0.5*f.y*f.y + 0.5*f.y*f.y*f.y;
            vec2 sample_uv = (c + vec2(offsets[x], offsets[y])) * u_texelSize;
            sum += texture(tex, sample_uv) * wx2 * wy2;
        }
    }
    return sum;
}

vec4 sampleTex(sampler2D tex, vec2 uv) { return u_bicubic ? textureBicubic(tex, uv) : texture(tex, uv); }

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0); vec3 c1 = vec3(0.05, 0.03, 0.1); vec3 c2 = vec3(0.3, 0.1, 0.3);
    vec3 c3 = vec3(0.6, 0.2, 0.4); vec3 c4 = vec3(0.9, 0.4, 0.3); vec3 c5 = vec3(1.0, 0.8, 0.5);
    if(t < 0.2) return mix(c0, c1, t/0.2); if(t < 0.4) return mix(c1, c2, (t-0.2)/0.2);
    if(t < 0.6) return mix(c2, c3, (t-0.4)/0.2); if(t < 0.8) return mix(c3, c4, (t-0.6)/0.2);
    if(t < 0.95) return mix(c4, c5, (t-0.8)/0.15); return mix(c5, vec3(1.0), (t-0.95)/0.05);
}

void main() {
    vec2 vel = sampleTex(u_velocity, v_uv).xy;
    vec4 dens = sampleTex(u_density, v_uv);
    float smoke = clamp(dens.x, 0.0, 1.0);
    float temp = clamp(dens.y, 0.0, 1.0);
    vec3 col = vec3(0.0);

    if (u_render_mode == 0) {
        vec2 noise_uv = v_uv * 15.0 + vel * 5.0;
        float n = fbm(noise_uv);
        smoke *= mix(1.0, n * 1.5, u_detail_strength);
        smoke = clamp(smoke, 0.0, 1.0);

        vec3 fire_col = mix(u_color_fire * 0.5, u_color_fire, temp);
        fire_col = mix(fire_col, u_color_hot, pow(temp, 3.0));
        
        vec3 bg = vec3(0.02, 0.02, 0.03);
        col = mix(bg, u_color_smoke, smoke);
        col += fire_col * smoothstep(0.05, 0.8, temp);
        
        vec2 light_dir = vec2(cos(u_light_angle), sin(u_light_angle));
        float transmittance = 1.0;
        float light_accum = 0.0;
        for(int i=0; i<20; i++) {
            vec2 sample_uv = v_uv + light_dir * 0.02 * float(i);
            sample_uv = clamp(sample_uv, vec2(0.0), vec2(1.0));
            float dens_val = sampleTex(u_density, sample_uv).x;
            transmittance *= exp(-dens_val * 0.3);
            light_accum += transmittance * dens_val * 0.1;
        }
        vec3 light_color = vec3(1.0, 0.7, 0.3) * light_accum * u_light_intensity;
        col += light_color * smoke; 
        
        vec3 bloom = texture(u_bloom, v_uv).rgb;
        col += bloom * 1.5;
        col = col / (col + vec3(1.0));
        col = pow(col, vec3(1.0/2.2)); 
    } else if (u_render_mode == 1) {
        col = magma(clamp(length(vel) * 0.2, 0.0, 1.0));
    } else if (u_render_mode == 2) {
        col = magma(clamp(sampleTex(u_pressure, v_uv).x * 2.0 + 0.5, 0.0, 1.0));
    } else if (u_render_mode == 3) {
        col = magma(clamp(sampleTex(u_vorticity, v_uv).x * 2.0 + 0.5, 0.0, 1.0));
    } else if (u_render_mode == 4) {
        col = magma(temp);
    } else if (u_render_mode == 5) {
        vec2 vel_w = vel * 0.1;
        float t = u_time * 2.0;
        float n = noise(v_uv * 50.0 + t);
        vec2 adv = v_uv + vel_w;
        n += noise(adv * 50.0 + t) * 0.5;
        adv += vel_w;
        n += noise(adv * 50.0 + t) * 0.25;
        col = vec3(n) * magma(clamp(length(vel) * 0.2, 0.0, 1.0));
    } else if (u_render_mode == 6) {
        float p = sampleTex(u_pressure, v_uv).x;
        float bands = fract(p * 10.0);
        float lines = smoothstep(0.0, 0.1, bands) * smoothstep(1.0, 0.9, bands);
        col = vec3(1.0) - lines;
    } else if (u_render_mode == 7) {
        float dL = sampleTex(u_density, v_uv - vec2(u_texelSize.x, 0.0)).x;
        float dR = sampleTex(u_density, v_uv + vec2(u_texelSize.x, 0.0)).x;
        float dB = sampleTex(u_density, v_uv - vec2(0.0, u_texelSize.y)).x;
        float dT = sampleTex(u_density, v_uv + vec2(0.0, u_texelSize.y)).x;
        vec2 grad = vec2(dR - dL, dT - dB) * 20.0;
        float schlieren = length(grad);
        col = magma(clamp(schlieren, 0.0, 1.0));
    } else if (u_render_mode == 8) {
        float grid_size = 20.0;
        vec2 grid_uv = v_uv * grid_size;
        vec2 cell = floor(grid_uv);
        vec2 local = fract(grid_uv) - 0.5; 
        
        vec2 vel_cell = texture(u_velocity, (cell + 0.5) / grid_size).xy;
        float len = length(vel_cell);
        
        float angle = atan(vel_cell.y, vel_cell.x);
        float c = cos(-angle), s = sin(-angle);
        vec2 r_local = vec2(c * local.x - s * local.y, s * local.x + c * local.y);
        
        float shaft = smoothstep(0.05, 0.0, abs(r_local.y)) * step(r_local.x, 0.2) * step(-0.3, r_local.x);
        float head1 = smoothstep(0.1, 0.0, abs(r_local.y - 0.1)) * step(0.2, r_local.x) * step(r_local.x, 0.3);
        float head2 = smoothstep(0.1, 0.0, abs(r_local.y + 0.1)) * step(0.2, r_local.x) * step(r_local.x, 0.3);
        
        float arrow = clamp(shaft + head1 + head2, 0.0, 1.0) * step(0.01, len);
        col = mix(col * 0.3, vec3(1.0), arrow);
    }
    
    if (u_render_mode > 0 && u_render_mode < 5) {
        vec2 bar_pos = vec2(0.85, 0.1); vec2 bar_size = vec2(0.03, 0.35); vec2 border = vec2(0.003);
        if (v_uv.x > bar_pos.x - border.x && v_uv.x < bar_pos.x + bar_size.x + border.x &&
            v_uv.y > bar_pos.y - border.y && v_uv.y < bar_pos.y + bar_size.y + border.y) {
            if (v_uv.x > bar_pos.x && v_uv.x < bar_pos.x + bar_size.x &&
                v_uv.y > bar_pos.y && v_uv.y < bar_pos.y + bar_size.y) {
                col = magma((v_uv.y - bar_pos.y) / bar_size.y);
            } else { col = vec3(0.8); }
        }
    }
    FragColor = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------
# 3D VOLUMETRIC SMOKE SHADERS
# ---------------------------------------------------------

FRAG_DISPLAY_3D = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_zoom;
uniform float u_pan_x;
uniform float u_pan_y;
uniform float u_density;
uniform float u_speed;
uniform int u_steps;
uniform float u_light_intensity;
uniform float u_pyroclastic;
uniform float u_scattering;
uniform float u_phase_g; // New anisotropy parameter
uniform bool u_dither;
uniform int u_render_mode;
uniform float u_exposure;
uniform float u_clip_plane;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1); p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise(vec3 p) {
    vec3 i = floor(p); vec3 f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i+vec3(0,0,0)), hash(i+vec3(1,0,0)), f.x),
                   mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                   mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float v = 0.0; float a = 0.5;
    for(int i=0; i<4; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}
vec3 getFlow(vec3 p) {
    return vec3(noise(p + vec3(0.0)) - 0.5, noise(p + vec3(100.0)) - 0.5, noise(p + vec3(0.0, 0.0, 100.0)) - 0.5) * 2.0;
}
mat3 rotY(float a) { float c=cos(a), s=sin(a); return mat3(c,0,s, 0,1,0, -s,0,c); }
mat3 rotX(float a) { float c=cos(a), s=sin(a); return mat3(1,0,0, 0,c,-s, 0,s,c); }

// Henyey-Greenstein Phase Function
float phase(float mu, float g) { 
    return (1.0 - g*g) / (4.0 * 3.14159 * pow(1.0 + g*g - 2.0*g*mu, 1.5)); 
}

vec3 magma3d(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0); vec3 c1 = vec3(0.05, 0.03, 0.1); vec3 c2 = vec3(0.3, 0.1, 0.3);
    vec3 c3 = vec3(0.6, 0.2, 0.4); vec3 c4 = vec3(0.9, 0.4, 0.3); vec3 c5 = vec3(1.0, 0.8, 0.5);
    if(t < 0.2) return mix(c0, c1, t/0.2); if(t < 0.4) return mix(c1, c2, (t-0.2)/0.2);
    if(t < 0.6) return mix(c2, c3, (t-0.4)/0.2); if(t < 0.8) return mix(c3, c4, (t-0.6)/0.2);
    if(t < 0.95) return mix(c4, c5, (t-0.8)/0.15); return mix(c5, vec3(1.0), (t-0.95)/0.05);
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float bnoise(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;
    
    mat3 rot = rotY(u_yaw) * rotX(u_pitch);
    mat3 inv_rot = transpose(rot);
    vec3 target = vec3(u_pan_x, u_pan_y, 0.0);
    vec3 ro = target + rot * vec3(0.0, 0.0, -u_zoom);
    vec3 rd = rot * normalize(vec3(uv, 1.5));
    
    float b = dot(ro - target, rd);
    float c = dot(ro - target, ro - target) - 2.25;
    float t = -1.0;
    if(b*b - c > 0.0) t = -b - sqrt(b*b - c);
    
    vec4 col = vec4(0.0);
    float transmittance = 1.0;
    float ground_shadow = 0.0;
    
    float t_ground = -(ro.y - 0.0) / rd.y; 
    if (rd.y < 0.0) {
        vec3 p_ground = ro + rd * t_ground;
        if (length(p_ground.xz) < 4.0) {
            for(int s=0; s<10; s++) {
                vec3 sp = vec3(p_ground.x, -1.5 + float(s) * 0.3, p_ground.z);
                vec3 sflow = getFlow(sp * 0.5 + u_time * 0.1) * u_speed;
                ground_shadow += smoothstep(0.4, 1.0, fbm(sp + sflow * u_time)) * u_density * 0.2;
            }
        }
    }
    
    if(t > 0.0) {
        vec3 light_dir = normalize(rot * vec3(1.0, 1.0, -1.0));
        float step_size = 3.0 / float(u_steps);
        if(u_dither) t += bnoise(gl_FragCoord.xy) * step_size;
        
        for(int i=0; i<u_steps; i++) {
            vec3 p = ro + rd * t;
            vec3 p_local = inv_rot * (p - target);
            
            if (p_local.y > u_clip_plane) {
                t += step_size;
                continue;
            }
            
            vec3 flow = getFlow(p * 0.5 + u_time * 0.1) * u_speed;
            vec3 p_adv = p + flow * u_time;
            
            float d = fbm(p_adv * 1.5);
            if(u_pyroclastic > 0.0) {
                float edge = smoothstep(0.3, 0.6, d) * (1.0 - smoothstep(0.6, 0.9, d));
                vec3 disp = vec3(noise(p_adv * 8.0), noise(p_adv * 8.0 + 10.0), noise(p_adv * 8.0 + 20.0)) - 0.5;
                d += edge * dot(disp, normalize(p)) * u_pyroclastic * 2.0;
            }
            d = smoothstep(0.4, 1.0, d) * u_density;
            
            if(d > 0.01) {
                // Volumetric Shadow Ray with Powder Effect
                float shadow_dens = 0.0;
                for(int s=0; s<10; s++) {
                    vec3 sp = p + light_dir * float(s) * 0.15;
                    vec3 sflow = getFlow(sp * 0.5 + u_time * 0.1) * u_speed;
                    shadow_dens += smoothstep(0.4, 1.0, fbm(sp + sflow * u_time)) * u_density * 0.2;
                }
                float shadow = exp(-shadow_dens * 3.0);
                
                // Powder Effect (darkens edges)
                float powder = 1.0 - exp(-d * 4.0);
                shadow *= mix(1.0, powder, 0.5);
                
                float mu = dot(rd, light_dir);
                float ph = phase(mu, u_phase_g); // Use dynamic anisotropy
                float ms = mix(1.0, 0.2, u_scattering);
                
                vec3 light_col = vec3(1.0, 0.6, 0.3) * shadow * ph * u_light_intensity;
                vec3 ambient = vec3(0.1, 0.1, 0.15) * ms;
                vec3 emit = vec3(0.0);
                
                if (u_render_mode == 0) {
                    emit = vec3(0.9, 0.3, 0.1) * smoothstep(0.7, 1.0, d) * 2.0; 
                } else {
                    emit = magma3d(d) * 2.0; 
                    light_col = vec3(0.0); 
                }
                
                col.rgb += (light_col + ambient + emit) * d * transmittance;
                transmittance *= exp(-d * 0.4);
            }
            if(transmittance < 0.01) break;
            t += step_size;
        }
    }
    
    if (rd.y < 0.0 && t_ground > 0.0) {
        vec3 p_ground = ro + rd * t_ground;
        vec2 grid_uv = fract(p_ground.xz * 2.0) - 0.5;
        float grid_line = smoothstep(0.48, 0.5, max(abs(grid_uv.x), abs(grid_uv.y)));
        vec3 ground_col = mix(vec3(0.05, 0.05, 0.08), vec3(0.1, 0.1, 0.12), grid_line);
        float shadow_factor = exp(-ground_shadow * 3.0);
        ground_col *= mix(0.2, 1.0, shadow_factor); 
        col.rgb = mix(col.rgb, ground_col, 0.8); 
    }
    
    vec3 bg = vec3(0.02, 0.02, 0.03);
    col.rgb += bg * (1.0 - (1.0 - transmittance));
    
    col.rgb = ACESFilm(col.rgb * u_exposure);
    col.rgb = pow(col.rgb, vec3(1.0/2.2));
    FragColor = vec4(col.rgb, 1.0);
}
"""

# ---------------------------------------------------------
# PYTHON OPENGL ENGINE
# ---------------------------------------------------------

class PingPong:
    def __init__(self, width, height, internal_format=GL_RGBA16F):
        self.width = width
        self.height = height
        self.fbo1 = self._create_fbo(internal_format)
        self.fbo2 = self._create_fbo(internal_format)
        self.read = self.fbo1
        self.write = self.fbo2

    def _create_fbo(self, internal_format):
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return fbo

    def swap(self): self.read, self.write = self.write, self.read

    def bind_texture(self, unit, fbo):
        tex_id = glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, fbo)
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        
    def clear(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo1); glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo2); glClear(GL_COLOR_BUFFER_BIT)

class FluidWidget2D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.SIM_SIZE = 256  
        self.timer = QElapsedTimer(); self.timer.start(); self.last_time = 0
        self.mouse_pos = None; self.last_mouse_pos = None
        self.left_mouse_down = False; self.right_mouse_down = False
        
        self.is_paused = False
        self.curl_strength = 40.0; self.vort_decay = 0.1
        self.wavelet_amp = 2.0 
        self.buoyancy = 1.5; self.weight = 0.3
        self.vel_dissipation = 0.99; self.smoke_dissipation = 0.97; self.temp_dissipation = 0.90
        
        self.auto_emit = True
        self.emit_x = 0.5
        self.emit_force = 0.8
        
        self.use_maccormack = True
        self.jacobi_iters = 20
        
        self.render_mode = 0
        self.light_intensity = 2.0; self.light_angle = 1.5708 
        self.detail_strength = 0.4
        self.bicubic = True
        self.show_particles = True; self.particle_size = 2.0
        
        self.presets = {
            "Classic Fire": [(0.08, 0.08, 0.1), (1.0, 0.4, 0.1), (1.0, 0.9, 0.7)],
            "Toxic Green": [(0.1, 0.1, 0.05), (0.2, 1.0, 0.1), (0.9, 1.0, 0.5)],
            "Ghostly Blue": [(0.05, 0.08, 0.1), (0.1, 0.4, 1.0), (0.7, 0.9, 1.0)],
            "Magenta Plasma": [(0.1, 0.05, 0.1), (0.9, 0.1, 0.8), (1.0, 0.7, 1.0)]
        }
        self.current_preset = "Classic Fire"
        self.qtimer = QTimer(self); self.qtimer.timeout.connect(self.update)
        self.frame_count = 0; self.fps_timer = QElapsedTimer(); self.fps_timer.start(); self.fps = 0
        
        self.setMouseTracking(True)

    def initializeGL(self):
        from OpenGL.GL import shaders
        def prog(v, f): return shaders.compileProgram(shaders.compileShader(v, GL_VERTEX_SHADER), shaders.compileShader(f, GL_FRAGMENT_SHADER))
        self.prog_splat = prog(VERT_SHADER, FRAG_SPLAT)
        self.prog_buoy = prog(VERT_SHADER, FRAG_BUOYANCY)
        self.prog_adv_mc = prog(VERT_SHADER, FRAG_ADVECT_MACCORMACK)
        self.prog_adv_sl = prog(VERT_SHADER, FRAG_ADVECT_SL)
        self.prog_div = prog(VERT_SHADER, FRAG_DIVERGENCE)
        self.prog_jacobi = prog(VERT_SHADER, FRAG_JACOBI)
        self.prog_grad = prog(VERT_SHADER, FRAG_GRADIENT_SUB)
        self.prog_bound = prog(VERT_SHADER, FRAG_BOUNDARY)
        self.prog_vort = prog(VERT_SHADER, FRAG_VORTICITY)
        self.prog_vort_force = prog(VERT_SHADER, FRAG_VORTICITY_FORCE)
        self.prog_wavelet = prog(VERT_SHADER, FRAG_WAVELET_TURBULENCE) 
        self.prog_noise = prog(VERT_SHADER, FRAG_CURL_NOISE)
        self.prog_bright = prog(VERT_SHADER, FRAG_BRIGHT_PASS)
        self.prog_blur = prog(VERT_SHADER, FRAG_BLUR)
        self.prog_display = prog(VERT_SHADER, FRAG_DISPLAY_2D)
        self.prog_part_update = prog(VERT_SHADER, FRAG_PARTICLE_UPDATE)
        self.prog_part_render = prog(VERT_PARTICLE, FRAG_PARTICLE)

        self.vao = glGenVertexArrays(1); self.vbo = glGenBuffers(1)
        verts = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo); glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindVertexArray(self.vao); glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.particle_vao = glGenVertexArrays(1)
        self._init_fbos(); glDisable(GL_DEPTH_TEST)

    def showEvent(self, event): self.qtimer.start(16)
    def hideEvent(self, event): self.qtimer.stop()

    def _init_fbos(self):
        self.vel = PingPong(self.SIM_SIZE, self.SIM_SIZE); self.dens = PingPong(self.SIM_SIZE, self.SIM_SIZE)
        self.press = PingPong(self.SIM_SIZE, self.SIM_SIZE); self.div = PingPong(self.SIM_SIZE, self.SIM_SIZE)
        self.vort = PingPong(self.SIM_SIZE, self.SIM_SIZE); self.noise = PingPong(self.SIM_SIZE, self.SIM_SIZE)
        self.bloom = PingPong(self.SIM_SIZE // 2, self.SIM_SIZE // 2)
        self.particles = PingPong(256, 256, GL_RGBA32F) 

    def clear_fields(self):
        for p in [self.vel, self.dens, self.press, self.div, self.vort, self.particles]: p.clear()

    def set_resolution(self, size): self.SIM_SIZE = size; self._init_fbos()

    def _draw_fullscreen(self): glBindVertexArray(self.vao); glDrawArrays(GL_TRIANGLES, 0, 3)

    def paintGL(self):
        current_time = self.timer.elapsed() / 1000.0
        dt = min(current_time - self.last_time, 0.033); self.last_time = current_time
        if dt == 0: dt = 0.016
        self.frame_count += 1
        if self.fps_timer.hasExpired(500):
            self.fps = int(self.frame_count * 1000.0 / self.fps_timer.restart()); self.frame_count = 0
            self.window().update_status(f"FPS: {self.fps} | Grid: {self.SIM_SIZE}x{self.SIM_SIZE} | Mode: {self.render_mode}")
        texel_size = (1.0 / self.SIM_SIZE, 1.0 / self.SIM_SIZE)
        if not self.is_paused: self._simulate(dt, current_time, texel_size)
        self._render(dt, current_time, texel_size)

    def _simulate(self, dt, current_time, texel_size):
        if self.left_mouse_down or self.right_mouse_down:
            dx = dy = 0.0
            if self.last_mouse_pos and self.mouse_pos:
                dx = (self.mouse_pos[0] - self.last_mouse_pos[0]) * 4.0; dy = (self.mouse_pos[1] - self.last_mouse_pos[1]) * 4.0
            glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write); glViewport(0, 0, self.SIM_SIZE, self.SIM_SIZE)
            glUseProgram(self.prog_splat); self.vel.bind_texture(0, self.vel.read)
            glUniform1i(glGetUniformLocation(self.prog_splat, "u_target"), 0)
            glUniform2f(glGetUniformLocation(self.prog_splat, "u_point"), *self.mouse_pos)
            glUniform1f(glGetUniformLocation(self.prog_splat, "u_radius"), 0.002)
            glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), dx, dy, 0.0, 1.0)
            self._draw_fullscreen(); self.vel.swap()
            glBindFramebuffer(GL_FRAMEBUFFER, self.dens.write); self.dens.bind_texture(0, self.dens.read)
            glUniform1i(glGetUniformLocation(self.prog_splat, "u_target"), 0)
            if self.left_mouse_down: glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), 1.0, 2.0, 0.0, 1.0)
            else: glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), 1.5, -1.0, 0.0, 1.0)
            self._draw_fullscreen(); self.dens.swap()

        if self.auto_emit:
            emit_pos = (self.emit_x, 0.05)
            glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write); glUseProgram(self.prog_splat)
            self.vel.bind_texture(0, self.vel.read)
            glUniform2f(glGetUniformLocation(self.prog_splat, "u_point"), *emit_pos)
            glUniform1f(glGetUniformLocation(self.prog_splat, "u_radius"), 0.005)
            glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), 0.0, self.emit_force, 0.0, 1.0)
            self._draw_fullscreen(); self.vel.swap()
            glBindFramebuffer(GL_FRAMEBUFFER, self.dens.write); self.dens.bind_texture(0, self.dens.read)
            glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), 0.6, 1.8, 0.0, 1.0)
            self._draw_fullscreen(); self.dens.swap()

        self.last_mouse_pos = self.mouse_pos

        glBindFramebuffer(GL_FRAMEBUFFER, self.noise.write); glViewport(0, 0, self.SIM_SIZE, self.SIM_SIZE)
        glUseProgram(self.prog_noise); glUniform1f(glGetUniformLocation(self.prog_noise, "u_time"), current_time)
        self._draw_fullscreen(); self.noise.swap()
        
        if self.auto_emit:
            glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write); glUseProgram(self.prog_splat)
            self.vel.bind_texture(0, self.vel.read)
            glUniform2f(glGetUniformLocation(self.prog_splat, "u_point"), self.emit_x, 0.05)
            glUniform1f(glGetUniformLocation(self.prog_splat, "u_radius"), 0.05)
            glUniform4f(glGetUniformLocation(self.prog_splat, "u_color"), 0.5, 0.5, 0.0, 1.0)
            self._draw_fullscreen(); self.vel.swap()

        glUseProgram(self.prog_vort); glUniform2f(glGetUniformLocation(self.prog_vort, "u_texelSize"), *texel_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vort.write); self.vel.bind_texture(0, self.vel.read)
        self._draw_fullscreen(); self.vort.swap()
        glUseProgram(self.prog_vort_force)
        glUniform2f(glGetUniformLocation(self.prog_vort_force, "u_texelSize"), *texel_size)
        glUniform1f(glGetUniformLocation(self.prog_vort_force, "u_curl"), self.curl_strength)
        glUniform1f(glGetUniformLocation(self.prog_vort_force, "u_decay"), self.vort_decay)
        glUniform1f(glGetUniformLocation(self.prog_vort_force, "u_dt"), dt)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write)
        self.vel.bind_texture(0, self.vel.read); self.vort.bind_texture(1, self.vort.read)
        self._draw_fullscreen(); self.vel.swap()

        if self.wavelet_amp > 0.0:
            glUseProgram(self.prog_wavelet)
            glUniform2f(glGetUniformLocation(self.prog_wavelet, "u_texelSize"), *texel_size)
            glUniform1f(glGetUniformLocation(self.prog_wavelet, "u_time"), current_time)
            glUniform1f(glGetUniformLocation(self.prog_wavelet, "u_amplitude"), self.wavelet_amp)
            glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write)
            self.vel.bind_texture(0, self.vel.read); self.vort.bind_texture(1, self.vort.read)
            self._draw_fullscreen(); self.vel.swap()

        glUseProgram(self.prog_buoy); glUniform1f(glGetUniformLocation(self.prog_buoy, "u_dt"), dt)
        glUniform1f(glGetUniformLocation(self.prog_buoy, "u_buoyancy"), self.buoyancy)
        glUniform1f(glGetUniformLocation(self.prog_buoy, "u_weight"), self.weight)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write)
        self.vel.bind_texture(0, self.vel.read); self.dens.bind_texture(1, self.dens.read)
        self._draw_fullscreen(); self.vel.swap()

        glUseProgram(self.prog_div); glUniform2f(glGetUniformLocation(self.prog_div, "u_texelSize"), *texel_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.div.write); self.vel.bind_texture(0, self.vel.read)
        self._draw_fullscreen(); self.div.swap()
        
        # Red-Black Gauss-Seidel Pressure Solve
        glUseProgram(self.prog_jacobi); glUniform2f(glGetUniformLocation(self.prog_jacobi, "u_texelSize"), *texel_size)
        for _ in range(self.jacobi_iters // 2):
            # Red Pass
            glUniform1i(glGetUniformLocation(self.prog_jacobi, "u_pass"), 0)
            glBindFramebuffer(GL_FRAMEBUFFER, self.press.write)
            self.press.bind_texture(0, self.press.read); self.div.bind_texture(1, self.div.read)
            self._draw_fullscreen(); self.press.swap()
            # Black Pass
            glUniform1i(glGetUniformLocation(self.prog_jacobi, "u_pass"), 1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.press.write)
            self.press.bind_texture(0, self.press.read); self.div.bind_texture(1, self.div.read)
            self._draw_fullscreen(); self.press.swap()
            
        glUseProgram(self.prog_grad); glUniform2f(glGetUniformLocation(self.prog_grad, "u_texelSize"), *texel_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write)
        self.vel.bind_texture(0, self.vel.read); self.press.bind_texture(1, self.press.read)
        self._draw_fullscreen(); self.vel.swap()

        glUseProgram(self.prog_bound); glUniform2f(glGetUniformLocation(self.prog_bound, "u_texelSize"), *texel_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write); self.vel.bind_texture(0, self.vel.read)
        self._draw_fullscreen(); self.vel.swap()

        adv_prog = self.prog_adv_mc if self.use_maccormack else self.prog_adv_sl
        glUseProgram(adv_prog); glUniform2f(glGetUniformLocation(adv_prog, "u_texelSize"), *texel_size)
        glUniform1f(glGetUniformLocation(adv_prog, "u_dt"), dt)
        
        # Advect Velocity
        glUniform3f(glGetUniformLocation(adv_prog, "u_dissipation"), self.vel_dissipation, 1.0, 1.0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.vel.write)
        self.vel.bind_texture(0, self.vel.read); self.vel.bind_texture(1, self.vel.read)
        self._draw_fullscreen(); self.vel.swap()
        
        # Advect Density & Temperature (Decoupled Dissipation)
        glUniform3f(glGetUniformLocation(adv_prog, "u_dissipation"), 1.0, self.smoke_dissipation, self.temp_dissipation)
        glBindFramebuffer(GL_FRAMEBUFFER, self.dens.write)
        self.vel.bind_texture(0, self.vel.read); self.dens.bind_texture(1, self.dens.read)
        self._draw_fullscreen(); self.dens.swap()

        glBindFramebuffer(GL_FRAMEBUFFER, self.particles.write); glViewport(0, 0, 256, 256)
        glUseProgram(self.prog_part_update)
        self.particles.bind_texture(0, self.particles.read); self.vel.bind_texture(1, self.vel.read)
        glUniform1i(glGetUniformLocation(self.prog_part_update, "u_particles"), 0)
        glUniform1i(glGetUniformLocation(self.prog_part_update, "u_velocity"), 1)
        glUniform1f(glGetUniformLocation(self.prog_part_update, "u_dt"), dt)
        glUniform1f(glGetUniformLocation(self.prog_part_update, "u_time"), current_time)
        self._draw_fullscreen(); self.particles.swap()

    def _render(self, dt, current_time, texel_size):
        bloom_size = (self.SIM_SIZE // 2, self.SIM_SIZE // 2); bloom_texel = (1.0 / bloom_size[0], 1.0 / bloom_size[1])
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom.write); glViewport(0, 0, bloom_size[0], bloom_size[1])
        glUseProgram(self.prog_bright); self.dens.bind_texture(0, self.dens.read); self._draw_fullscreen(); self.bloom.swap()
        glUseProgram(self.prog_blur); glUniform2f(glGetUniformLocation(self.prog_blur, "u_texelSize"), *bloom_texel)
        glUniform1i(glGetUniformLocation(self.prog_blur, "u_horizontal"), 1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom.write); self.bloom.bind_texture(0, self.bloom.read)
        self._draw_fullscreen(); self.bloom.swap()
        glUniform1i(glGetUniformLocation(self.prog_blur, "u_horizontal"), 0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom.write); self.bloom.bind_texture(0, self.bloom.read)
        self._draw_fullscreen(); self.bloom.swap()

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glViewport(0, 0, self.width(), self.height()); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.prog_display)
        self.dens.bind_texture(0, self.dens.read); self.bloom.bind_texture(1, self.bloom.read)
        self.press.bind_texture(2, self.press.read); self.vort.bind_texture(3, self.vort.read)
        self.vel.bind_texture(4, self.vel.read)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_density"), 0)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_bloom"), 1)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_pressure"), 2)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_vorticity"), 3)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_velocity"), 4)
        colors = self.presets[self.current_preset]
        glUniform3f(glGetUniformLocation(self.prog_display, "u_color_smoke"), *colors[0])
        glUniform3f(glGetUniformLocation(self.prog_display, "u_color_fire"), *colors[1])
        glUniform3f(glGetUniformLocation(self.prog_display, "u_color_hot"), *colors[2])
        glUniform1i(glGetUniformLocation(self.prog_display, "u_render_mode"), self.render_mode)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_light_intensity"), self.light_intensity)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_light_angle"), self.light_angle)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_detail_strength"), self.detail_strength)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_bicubic"), self.bicubic)
        glUniform2f(glGetUniformLocation(self.prog_display, "u_texelSize"), *texel_size)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_time"), current_time)
        self._draw_fullscreen()

        if self.show_particles and self.render_mode == 0:
            glUseProgram(self.prog_part_render)
            self.particles.bind_texture(0, self.particles.read)
            glUniform1i(glGetUniformLocation(self.prog_part_render, "u_particles"), 0)
            glUniform1f(glGetUniformLocation(self.prog_part_render, "u_size"), self.particle_size)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glBindVertexArray(self.particle_vao); glDrawArrays(GL_POINTS, 0, 256 * 256)
            glDisable(GL_BLEND)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.left_mouse_down = True; self.last_mouse_pos = None; self.mouse_pos = self._map_to_sim(event.position())
        elif event.button() == Qt.RightButton:
            self.right_mouse_down = True; self.last_mouse_pos = None; self.mouse_pos = self._map_to_sim(event.position())

    def mouseMoveEvent(self, event: QMouseEvent):
        uv = self._map_to_sim(event.position())
        self.window().update_status(f"UV: ({uv[0]:.2f}, {uv[1]:.2f})")
        if self.left_mouse_down or self.right_mouse_down:
            self.last_mouse_pos = self.mouse_pos; self.mouse_pos = uv

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton: self.left_mouse_down = False
        elif event.button() == Qt.RightButton: self.right_mouse_down = False

    def _map_to_sim(self, pos):
        return (pos.x() / self.width(), 1.0 - (pos.y() / self.height()))

# ---------------------------------------------------------
# 3D VOLUMETRIC WIDGET
# ---------------------------------------------------------

class SmokeWidget3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QElapsedTimer(); self.timer.start()
        self.yaw = 0.5; self.pitch = 0.2; self.zoom = 4.0
        self.pan_x = 0.0; self.pan_y = 0.0
        self.density = 1.5; self.speed = 1.0; self.steps = 64
        self.light_intensity = 3.0
        self.pyroclastic = 0.5
        self.scattering = 0.5
        self.phase_g = 0.6
        self.dither = True
        self.render_mode = 0
        self.exposure = 1.0
        self.auto_rotate = False
        self.clip_plane = 2.0
        self.last_mouse_pos = None; self.left_mouse_down = False; self.right_mouse_down = False
        self.qtimer = QTimer(self); self.qtimer.timeout.connect(self.update)
        self.setMouseTracking(True)

    def initializeGL(self):
        from OpenGL.GL import shaders
        self.prog_display = shaders.compileProgram(
            shaders.compileShader(VERT_SHADER, GL_VERTEX_SHADER),
            shaders.compileShader(FRAG_DISPLAY_3D, GL_FRAGMENT_SHADER)
        )
        self.vao = glGenVertexArrays(1); self.vbo = glGenBuffers(1)
        verts = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo); glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindVertexArray(self.vao); glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    def showEvent(self, event): self.qtimer.start(16)
    def hideEvent(self, event): self.qtimer.stop()

    def paintGL(self):
        current_time = self.timer.elapsed() / 1000.0
        if self.auto_rotate: self.yaw += 0.005
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glViewport(0, 0, self.width(), self.height()); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.prog_display)
        glUniform2f(glGetUniformLocation(self.prog_display, "u_resolution"), self.width(), self.height())
        glUniform1f(glGetUniformLocation(self.prog_display, "u_time"), current_time)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_yaw"), self.yaw)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_pitch"), self.pitch)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_zoom"), self.zoom)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_pan_x"), self.pan_x)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_pan_y"), self.pan_y)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_speed"), self.speed)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_steps"), self.steps)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_light_intensity"), self.light_intensity)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_pyroclastic"), self.pyroclastic)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_scattering"), self.scattering)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_phase_g"), self.phase_g)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_dither"), self.dither)
        glUniform1i(glGetUniformLocation(self.prog_display, "u_render_mode"), self.render_mode)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_exposure"), self.exposure)
        glUniform1f(glGetUniformLocation(self.prog_display, "u_clip_plane"), self.clip_plane)
        glBindVertexArray(self.vao); glDrawArrays(GL_TRIANGLES, 0, 3)
        
        self.window().update_status(f"Yaw: {self.yaw:.2f} | Pitch: {self.pitch:.2f} | Zoom: {self.zoom:.1f}")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton: self.left_mouse_down = True; self.last_mouse_pos = event.position()
        elif event.button() == Qt.RightButton: self.right_mouse_down = True; self.last_mouse_pos = event.position()
        
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.left_mouse_down and self.last_mouse_pos:
            dx = event.position().x() - self.last_mouse_pos.x(); dy = event.position().y() - self.last_mouse_pos.y()
            self.yaw += dx * 0.01; self.pitch += dy * 0.01
            self.pitch = max(-1.5, min(1.5, self.pitch)); self.last_mouse_pos = event.position()
        elif self.right_mouse_down and self.last_mouse_pos:
            dx = event.position().x() - self.last_mouse_pos.x(); dy = event.position().y() - self.last_mouse_pos.y()
            self.pan_x += dx * 0.01; self.pan_y -= dy * 0.01
            self.last_mouse_pos = event.position()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton: self.left_mouse_down = False
        elif event.button() == Qt.RightButton: self.right_mouse_down = False
        
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() / 120
        self.zoom = max(2.5, min(10.0, self.zoom - delta * 0.5))

# ---------------------------------------------------------
# CUSTOM GUI WIDGETS & STYLES
# ---------------------------------------------------------

QSS = """
QMainWindow { background-color: #1a1a2e; }
QLabel { color: #c2c2c2; font-size: 12px; }
QGroupBox {
    border: 1px solid #2e2e54;
    border-radius: 4px;
    margin-top: 14px;
    padding: 14px 8px 8px 8px;
    color: #8a8aa3;
    font-weight: bold;
    background-color: #16162a;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QSlider::groove:horizontal {
    border: 1px solid #2e2e54;
    height: 4px;
    background: #12122a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4a4ae0;
    border: none;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #7a7af0; }
QPushButton {
    background-color: #2e2e54;
    color: #c2c2c2;
    border: 1px solid #4a4ae0;
    border-radius: 3px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover { background-color: #4a4ae0; color: white; }
QComboBox {
    background-color: #2e2e54;
    color: #c2c2c2;
    border: 1px solid #4a4ae0;
    border-radius: 3px;
    padding: 4px;
}
QCheckBox { color: #c2c2c2; }
QTabWidget::pane { border: 1px solid #2e2e54; }
QTabBar::tab {
    background-color: #12122a;
    color: #8a8aa3;
    padding: 8px 16px;
    border: 1px solid #2e2e54;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #2e2e54; color: #4a4ae0; }
QDockWidget { border: 1px solid #2e2e54; }
QDockWidget::title { text-align: left; background-color: #12122a; padding: 6px 10px; color: #4a4ae0; font-weight: bold; }
QToolBar { background-color: #12122a; border-bottom: 1px solid #2e2e54; spacing: 4px; padding: 4px; }
QToolBar QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 3px; padding: 6px; color: #c2c2c2; }
QToolBar QToolButton:hover { background-color: #2e2e54; border: 1px solid #4a4ae0; }
QStatusBar { background-color: #12122a; color: #8a8aa3; border-top: 1px solid #2e2e54; }
QStatusBar QLabel { margin-left: 10px; }
QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical { border: none; background: #12122a; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4ae0; min-height: 20px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0; }
"""

class LabeledSlider(QWidget):
    def __init__(self, name, mn, mx, val, callback, decimals=1, suffix=""):
        super().__init__()
        self.callback = callback
        self.decimals = decimals
        self.suffix = suffix
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.label_name = QLabel(name)
        self.label_name.setMinimumWidth(110)
        self.label_name.setToolTip(f"Adjust the {name} parameter.")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(mn)
        self.slider.setMaximum(mx)
        self.slider.setValue(val)
        self.slider.setToolTip(f"Range: {mn/10**decimals} to {mx/10**decimals}")
        
        self.label_val = QLabel()
        self.label_val.setMinimumWidth(50)
        self.label_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label_val.setStyleSheet("color: #4a4ae0; font-weight: bold;")
        
        self.slider.valueChanged.connect(self.on_change)
        layout.addWidget(self.label_name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label_val)
        self.on_change(val)

    def on_change(self, v):
        val = v / (10 ** self.decimals)
        self.label_val.setText(f"{val:.{self.decimals}f}{self.suffix}")
        self.callback(val)

class ControlPanel2D(QWidget):
    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout()
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128", "256", "512"])
        self.res_combo.setCurrentText("256")
        self.res_combo.currentTextChanged.connect(lambda t: self.gl_widget.set_resolution(int(t)))
        sim_layout.addWidget(QLabel("Grid Resolution"))
        sim_layout.addWidget(self.res_combo)
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        num_group = QGroupBox("Numerical Methods")
        num_layout = QVBoxLayout()
        self.chk_maccormack = QCheckBox("MacCormack Advection")
        self.chk_maccormack.setChecked(True)
        self.chk_maccormack.stateChanged.connect(self.update_maccormack)
        self.chk_maccormack.setToolTip("Toggle between MacCormack (sharp) and Semi-Lagrangian (stable)")
        num_layout.addWidget(self.chk_maccormack)
        self.slider_jacobi = LabeledSlider("RBGS Iterations", 5, 40, self.gl_widget.jacobi_iters, self.update_jacobi, decimals=0)
        self.slider_jacobi.slider.setToolTip("Red-Black Gauss-Seidel solver iterations. Halved cost vs Jacobi.")
        num_layout.addWidget(self.slider_jacobi)
        num_group.setLayout(num_layout)
        layout.addWidget(num_group)

        emit_group = QGroupBox("Auto-Emitter Controls")
        emit_layout = QVBoxLayout()
        self.chk_emit = QCheckBox("Enable Auto-Emitter")
        self.chk_emit.setChecked(True)
        self.chk_emit.stateChanged.connect(self.update_emit)
        emit_layout.addWidget(self.chk_emit)
        self.slider_ex = LabeledSlider("Emitter X Pos", 0, 100, int(self.gl_widget.emit_x * 100), self.update_ex, decimals=2)
        emit_layout.addWidget(self.slider_ex)
        self.slider_eforce = LabeledSlider("Emitter Force", 0, 50, int(self.gl_widget.emit_force * 10), self.update_eforce)
        emit_layout.addWidget(self.slider_eforce)
        emit_group.setLayout(emit_layout)
        layout.addWidget(emit_group)

        viz_group = QGroupBox("Visualization Mode")
        viz_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["0: Cinematic", "1: Velocity", "2: Pressure", "3: Vorticity", "4: Temperature", "5: Streamlines", "6: Iso-Contours", "7: Schlieren", "8: Vector Glyphs"])
        self.mode_combo.currentIndexChanged.connect(self.update_mode)
        viz_layout.addWidget(self.mode_combo)
        self.chk_bicubic = QCheckBox("Bicubic Filtering")
        self.chk_bicubic.setChecked(True)
        self.chk_bicubic.stateChanged.connect(self.update_bicubic)
        viz_layout.addWidget(self.chk_bicubic)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        part_group = QGroupBox("Lagrangian Particles")
        part_layout = QVBoxLayout()
        self.chk_particles = QCheckBox("Show Particles")
        self.chk_particles.setChecked(True)
        self.chk_particles.stateChanged.connect(self.update_particles)
        part_layout.addWidget(self.chk_particles)
        self.slider_psize = LabeledSlider("Size", 1, 10, int(self.gl_widget.particle_size), self.update_psize, decimals=0)
        part_layout.addWidget(self.slider_psize)
        part_group.setLayout(part_layout)
        layout.addWidget(part_group)

        light_group = QGroupBox("Cinematic Lighting & Detail")
        light_layout = QVBoxLayout()
        self.slider_int = LabeledSlider("Intensity", 0, 100, int(self.gl_widget.light_intensity * 10), self.update_light_int)
        light_layout.addWidget(self.slider_int)
        self.slider_ang = LabeledSlider("Angle", 0, 360, int(np.degrees(self.gl_widget.light_angle)), self.update_light_ang, decimals=0, suffix="°")
        light_layout.addWidget(self.slider_ang)
        self.slider_detail = LabeledSlider("fBm Detail", 0, 100, int(self.gl_widget.detail_strength * 100), self.update_detail, decimals=2)
        light_layout.addWidget(self.slider_detail)
        light_group.setLayout(light_layout)
        layout.addWidget(light_group)

        phys_group = QGroupBox("SOTA Physics (2D)")
        phys_layout = QVBoxLayout()
        self.slider_wavelet = LabeledSlider("Wavelet Turb", 0, 50, int(self.gl_widget.wavelet_amp * 10), self.update_wavelet)
        phys_layout.addWidget(self.slider_wavelet)
        self.slider_curl = LabeledSlider("Vorticity", 0, 100, int(self.gl_widget.curl_strength), self.update_curl, decimals=0)
        phys_layout.addWidget(self.slider_curl)
        self.slider_decay = LabeledSlider("Vort Decay", 0, 100, int(self.gl_widget.vort_decay * 100), self.update_decay, decimals=2)
        phys_layout.addWidget(self.slider_decay)
        self.slider_buoy = LabeledSlider("Buoyancy", 0, 50, int(self.gl_widget.buoyancy * 10), self.update_buoy)
        phys_layout.addWidget(self.slider_buoy)
        self.slider_weight = LabeledSlider("Smoke Weight", 0, 50, int(self.gl_widget.weight * 100), self.update_weight, decimals=2)
        phys_layout.addWidget(self.slider_weight)
        self.slider_vel_diss = LabeledSlider("Vel Dissipation", 90, 100, int(self.gl_widget.vel_dissipation * 100), self.update_vel_diss, decimals=2)
        phys_layout.addWidget(self.slider_vel_diss)
        self.slider_smoke_diss = LabeledSlider("Smoke Dissipation", 90, 100, int(self.gl_widget.smoke_dissipation * 100), self.update_smoke_diss, decimals=2)
        phys_layout.addWidget(self.slider_smoke_diss)
        self.slider_temp_diss = LabeledSlider("Heat Dissipation", 50, 100, int(self.gl_widget.temp_dissipation * 100), self.update_temp_diss, decimals=2)
        phys_layout.addWidget(self.slider_temp_diss)
        phys_group.setLayout(phys_layout)
        layout.addWidget(phys_group)

        color_group = QGroupBox("Color Preset")
        color_layout = QVBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.addItems(self.gl_widget.presets.keys())
        self.color_combo.currentTextChanged.connect(self.update_colors)
        color_layout.addWidget(self.color_combo)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        lbl_help = QLabel("L-Click: Fire\nR-Click: Smoke\nKeys: Space(C) 1-9")
        lbl_help.setStyleSheet("color: #5a5a7a; font-size: 11px;")
        layout.addWidget(lbl_help)

    def update_maccormack(self, s): self.gl_widget.use_maccormack = bool(s)
    def update_jacobi(self, v): self.gl_widget.jacobi_iters = int(v)
    def update_emit(self, s): self.gl_widget.auto_emit = bool(s)
    def update_ex(self, v): self.gl_widget.emit_x = v
    def update_eforce(self, v): self.gl_widget.emit_force = v
    def update_mode(self, m): self.gl_widget.render_mode = m
    def update_bicubic(self, s): self.gl_widget.bicubic = bool(s)
    def update_particles(self, s): self.gl_widget.show_particles = bool(s)
    def update_psize(self, v): self.gl_widget.particle_size = float(v)
    def update_light_int(self, v): self.gl_widget.light_intensity = v
    def update_light_ang(self, v): self.gl_widget.light_angle = np.radians(v)
    def update_detail(self, v): self.gl_widget.detail_strength = v
    def update_wavelet(self, v): self.gl_widget.wavelet_amp = v
    def update_curl(self, v): self.gl_widget.curl_strength = float(v)
    def update_decay(self, v): self.gl_widget.vort_decay = v
    def update_buoy(self, v): self.gl_widget.buoyancy = v
    def update_weight(self, v): self.gl_widget.weight = v
    def update_vel_diss(self, v): self.gl_widget.vel_dissipation = v
    def update_smoke_diss(self, v): self.gl_widget.smoke_dissipation = v
    def update_temp_diss(self, v): self.gl_widget.temp_dissipation = v
    def update_colors(self, text): self.gl_widget.current_preset = text

class ControlPanel3D(QWidget):
    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        info_group = QGroupBox("3D Volumetric Raymarching")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Renders true 3D procedural smoke\nusing SOTA multiple scattering,\npyroclastic displacement, and\nblue-noise ray dithering."))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        cam_group = QGroupBox("Camera & Tone Mapping")
        cam_layout = QVBoxLayout()
        self.chk_rotate = QCheckBox("Auto-Rotate")
        self.chk_rotate.stateChanged.connect(self.update_rotate)
        cam_layout.addWidget(self.chk_rotate)
        self.slider_exp = LabeledSlider("Exposure", 0, 30, int(self.gl_widget.exposure * 10), self.update_exp)
        cam_layout.addWidget(self.slider_exp)
        self.slider_clip = LabeledSlider("Clipping Plane", 0, 30, int(self.gl_widget.clip_plane * 10), self.update_clip)
        cam_layout.addWidget(self.slider_clip)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        param_group = QGroupBox("SOTA Rendering (3D)")
        param_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["0: Cinematic Fire", "1: Scientific Magma"])
        self.mode_combo.currentIndexChanged.connect(self.update_mode)
        param_layout.addWidget(QLabel("Render Mode"))
        param_layout.addWidget(self.mode_combo)
        
        self.slider_dens = LabeledSlider("Density", 0, 50, int(self.gl_widget.density * 10), self.update_dens)
        param_layout.addWidget(self.slider_dens)
        self.slider_speed = LabeledSlider("Flow Speed", 0, 50, int(self.gl_widget.speed * 10), self.update_speed)
        param_layout.addWidget(self.slider_speed)
        self.slider_steps = LabeledSlider("Ray Steps", 16, 128, int(self.gl_widget.steps), self.update_steps, decimals=0)
        param_layout.addWidget(self.slider_steps)
        self.slider_light = LabeledSlider("Light Intensity", 0, 100, int(self.gl_widget.light_intensity * 10), self.update_light)
        param_layout.addWidget(self.slider_light)
        self.slider_pyro = LabeledSlider("Pyroclastic Detail", 0, 100, int(self.gl_widget.pyroclastic * 100), self.update_pyro, decimals=2)
        param_layout.addWidget(self.slider_pyro)
        self.slider_scat = LabeledSlider("Multiple Scattering", 0, 100, int(self.gl_widget.scattering * 100), self.update_scat, decimals=2)
        param_layout.addWidget(self.slider_scat)
        self.slider_phase = LabeledSlider("Phase Anisotropy", -90, 90, int(self.gl_widget.phase_g * 100), self.update_phase, decimals=2)
        param_layout.addWidget(self.slider_phase)
        self.chk_dither = QCheckBox("Blue-Noise Dithering")
        self.chk_dither.setChecked(True)
        self.chk_dither.stateChanged.connect(self.update_dither)
        param_layout.addWidget(self.chk_dither)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        lbl_help = QLabel("L-Click: Orbit\nR-Click: Pan\nScroll: Zoom")
        lbl_help.setStyleSheet("color: #5a5a7a; font-size: 11px;")
        layout.addWidget(lbl_help)

    def update_rotate(self, s): self.gl_widget.auto_rotate = bool(s)
    def update_exp(self, v): self.gl_widget.exposure = v
    def update_clip(self, v): self.gl_widget.clip_plane = v
    def update_mode(self, m): self.gl_widget.render_mode = m
    def update_dens(self, v): self.gl_widget.density = v
    def update_speed(self, v): self.gl_widget.speed = v
    def update_steps(self, v): self.gl_widget.steps = int(v)
    def update_light(self, v): self.gl_widget.light_intensity = v
    def update_pyro(self, v): self.gl_widget.pyroclastic = v
    def update_scat(self, v): self.gl_widget.scattering = v
    def update_phase(self, v): self.gl_widget.phase_g = v
    def update_dither(self, s): self.gl_widget.dither = bool(s)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOTA GPU Fluid Solver - Professional VFX & CFD Tool")
        self.resize(1280, 800)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
        
        self.setStyleSheet(QSS)

        # Central Tab Widget
        self.tabs = QTabWidget()
        self.gl_2d = FluidWidget2D()
        self.gl_3d = SmokeWidget3D()
        self.tabs.addTab(self.gl_2d, "2D Physics Solver")
        self.tabs.addTab(self.gl_3d, "3D Volumetric Smoke")
        self.setCentralWidget(self.tabs)

        # Top Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        self.act_pause = QAction("Pause", self)
        self.act_pause.setCheckable(True)
        self.act_pause.setShortcut("Space")
        self.act_pause.triggered.connect(self.toggle_pause)
        self.toolbar.addAction(self.act_pause)

        self.act_clear = QAction("Clear Fields", self)
        self.act_clear.setShortcut("C")
        self.act_clear.triggered.connect(self.clear_fields)
        self.toolbar.addAction(self.act_clear)

        self.toolbar.addSeparator()

        self.act_reset_cam = QAction("Reset 3D Camera", self)
        self.act_reset_cam.setShortcut("R")
        self.act_reset_cam.triggered.connect(self.reset_3d_cam)
        self.toolbar.addAction(self.act_reset_cam)

        self.toolbar.addSeparator()

        self.act_screenshot = QAction("Save Screenshot", self)
        self.act_screenshot.triggered.connect(self.save_screenshot)
        self.toolbar.addAction(self.act_screenshot)

        # Dockable Right Panel
        self.dock = QDockWidget("Control Panel", self)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Scroll Area for Dock
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.panels = QStackedWidget()
        self.panel_2d = ControlPanel2D(self.gl_2d)
        self.panel_3d = ControlPanel3D(self.gl_3d)
        self.panels.addWidget(self.panel_2d)
        self.panels.addWidget(self.panel_3d)
        
        scroll.setWidget(self.panels)
        self.dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        
        self.tabs.currentChanged.connect(self.panels.setCurrentIndex)

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready.")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() >= Qt.Key_1 and event.key() <= Qt.Key_9:
            if self.tabs.currentIndex() == 0:
                mode = event.key() - Qt.Key_1
                if mode < self.panel_2d.mode_combo.count():
                    self.panel_2d.mode_combo.setCurrentIndex(mode)
        super().keyPressEvent(event)

    def toggle_pause(self):
        is_paused = self.act_pause.isChecked()
        self.gl_2d.is_paused = is_paused
        self.act_pause.setText("Resume" if is_paused else "Pause")

    def clear_fields(self):
        if self.tabs.currentIndex() == 0:
            self.gl_2d.clear_fields()

    def reset_3d_cam(self):
        self.gl_3d.yaw = 0.5
        self.gl_3d.pitch = 0.2
        self.gl_3d.zoom = 4.0
        self.gl_3d.pan_x = 0.0
        self.gl_3d.pan_y = 0.0

    def save_screenshot(self):
        widget = self.gl_2d if self.tabs.currentIndex() == 0 else self.gl_3d
        img = widget.grabFramebuffer()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Images (*.png)")
        if file_path:
            img.save(file_path)

    def update_status(self, msg):
        self.status.showMessage(msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())