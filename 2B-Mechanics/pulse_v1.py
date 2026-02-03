import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ────────────────────────────────────────────────
# Constants (Earth)
# ────────────────────────────────────────────────
MU_EARTH    = 3.986004418e14      # m³/s²
R_EARTH     = 6371000             # mean radius (m)
ALTITUDE    = 400000              # example LEO altitude (m)
ORBIT_TIME  = 90 * 60             # rough orbital period ~90 min (s)

# ────────────────────────────────────────────────
# Two-body equations of motion
# ────────────────────────────────────────────────
def two_body_ode(t, y, mu):
    r = y[0:3]
    v = y[3:6]
    r_norm = np.linalg.norm(r)
    accel = -mu * r / r_norm**3
    return np.concatenate((v, accel))

# ────────────────────────────────────────────────
# Setup initial conditions – circular equatorial LEO
# ────────────────────────────────────────────────
r0 = np.array([R_EARTH + ALTITUDE, 0.0, 0.0])
v_circ = np.sqrt(MU_EARTH / np.linalg.norm(r0))
v0 = np.array([0.0, v_circ, 0.0])

y0 = np.concatenate((r0, v0))

# Simulate one full orbit + a bit extra
t_span = (0, ORBIT_TIME * 1.1)
t_eval = np.linspace(0, t_span[1], 1200)   # more points = smoother animation

sol = solve_ivp(
    fun=two_body_ode,
    t_span=t_span,
    y0=y0,
    args=(MU_EARTH,),
    method='RK45',
    rtol=1e-9, atol=1e-9,
    t_eval=t_eval
)

positions = sol.y[0:3, :].T          # shape: (n_points, 3)

# ────────────────────────────────────────────────
# Animation setup
# ────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')            # space background

# Earth – better looking sphere (blue with simple grid)
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 60)
x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
z_earth = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

earth = ax.plot_surface(
    x_earth, y_earth, z_earth,
    color='royalblue', alpha=0.7, zorder=1,
    rstride=2, cstride=2, linewidth=0
)

# Add faint grid lines on Earth
ax.plot_wireframe(
    x_earth, y_earth, z_earth,
    color='lightblue', alpha=0.15, rstride=6, cstride=6
)

# Orbit trail (full path, semi-transparent)
trail, = ax.plot(
    positions[:,0], positions[:,1], positions[:,2],
    color='cyan', linewidth=1.2, alpha=0.4, zorder=3
)

# Moving satellite (glowing point)
satellite, = ax.plot(
    [positions[0,0]], [positions[0,1]], [positions[0,2]],
    'o', color='lime', markersize=10,
    markeredgecolor='white', markeredgewidth=1.5,
    zorder=10
)

# Set equal aspect and limits
max_r = np.max(np.linalg.norm(positions, axis=1)) * 1.15
ax.set_xlim(-max_r, max_r)
ax.set_ylim(-max_r, max_r)
ax.set_zlim(-max_r, max_r)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f"Animated LEO Orbit @ {ALTITUDE//1000} km altitude", color='white')
ax.tick_params(colors='lightgray')

# Make background fully space-like
fig.patch.set_facecolor('black')
ax.grid(False)

# Animation update function
def update(frame):
    satellite.set_data_3d(
        [positions[frame,0]],
        [positions[frame,1]],
        [positions[frame,2]]
    )
    return satellite,

# Create animation
ani = animation.FuncAnimation(
    fig, update,
    frames=len(t_eval),
    interval=30,           # ms between frames → ~33 fps
    blit=False,
    repeat=True
)

ax.view_init(elev=20, azim=30)   # nice starting camera angle
plt.show()

# Optional: save as gif (uncomment if you have pillow or imagemagick)
# ani.save('orbit_animation.gif', writer='pillow', fps=30, dpi=100)