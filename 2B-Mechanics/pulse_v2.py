import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk, messagebox


# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
MU_EARTH = 3.986004418e14      # m³/s²
R_EARTH  = 6371000.0           # mean radius (m)


# ────────────────────────────────────────────────
# Preset orbits (name, a_km, e, i_deg, Omega_deg, omega_deg, nu_deg)
# ────────────────────────────────────────────────
PRESETS = [
    ("1. Circular LEO (equatorial)",      6771,  0.00,   0.0,   0.0,   0.0,   0.0),
    ("2. Circular LEO (28.5° incl.)",     6771,  0.00,  28.5,   0.0,   0.0,   0.0),
    ("3. Circular Polar LEO",             7171,  0.00,  90.0,   0.0,   0.0,   0.0),
    ("4. Sun-Synchronous ~98°",           7071,  0.00,  98.0,   0.0,   0.0,   0.0),
    ("5. Molniya-type Highly Elliptical", 26554, 0.72,  63.4,   0.0, 270.0,   0.0),
    ("6. Geostationary Equatorial",      42164,  0.00,   0.0,   0.0,   0.0,   0.0),
    ("7. Geosynchronous Inclined 30°",   42164,  0.00,  30.0,   0.0,   0.0,   0.0),
    ("8. GPS-like MEO",                  26560,  0.00,  55.0,   0.0,   0.0,   0.0),
    ("9. Slightly Elliptical LEO",        6771,  0.05,  51.6,   0.0,   0.0,   0.0),
    ("10. GTO-like Transfer Orbit",      24000,  0.70,  28.5,   0.0, 180.0,   0.0),
]


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
# Convert classical orbital elements to position & velocity
# ────────────────────────────────────────────────
def kepler_to_cartesian(a, e, i, Omega, omega, nu, mu):
    p = a * (1 - e**2)                      # semi-latus rectum
    r = p / (1 + e * np.cos(nu))            # radius at true anomaly
    
    # Position vector in perifocal frame
    r_peri = np.array([
        r * np.cos(nu),
        r * np.sin(nu),
        0.0
    ])
    
    # Velocity vector in perifocal frame
    sqrt_mu_p = np.sqrt(mu / p)
    v_peri = np.array([
        -sqrt_mu_p * np.sin(nu),
        sqrt_mu_p * (e + np.cos(nu)),
        0.0
    ])
    
    # Rotation matrices
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    
    # Total rotation: Ω → i → ω
    R = rot_z(Omega) @ rot_x(i) @ rot_z(omega)
    
    r_inertial = R @ r_peri
    v_inertial = R @ v_peri
    
    return r_inertial, v_inertial


# ────────────────────────────────────────────────
# Run simulation and show animation
# ────────────────────────────────────────────────
def run_simulation(a_km, e, i_deg, Omega_deg, omega_deg, nu_deg, duration_orbits):
    try:
        a = float(a_km) * 1000
        e_val = float(e)
        i_rad = np.deg2rad(float(i_deg))
        Omega_rad = np.deg2rad(float(Omega_deg))
        omega_rad = np.deg2rad(float(omega_deg))
        nu_rad = np.deg2rad(float(nu_deg))
        orbits = float(duration_orbits)

        if e_val < 0 or e_val >= 1:
            raise ValueError("Eccentricity must be 0 ≤ e < 1 for closed orbits")
        if a < R_EARTH * 1.01:
            raise ValueError("Semi-major axis too small (inside Earth)")

        r0, v0 = kepler_to_cartesian(a, e_val, i_rad, Omega_rad, omega_rad, nu_rad, MU_EARTH)
        y0 = np.concatenate((r0, v0))

        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        t_total = period * orbits * 1.05
        t_eval = np.linspace(0, t_total, 1200)

        sol = solve_ivp(two_body_ode, (0, t_total), y0, args=(MU_EARTH,),
                        method='RK45', rtol=1e-9, atol=1e-9, t_eval=t_eval)

        positions = sol.y[0:3, :].T

        # ── Animation ───────────────────────────────────────
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')

        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 60)
        x_e = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y_e = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z_e = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x_e, y_e, z_e, color='royalblue', alpha=0.7)
        ax.plot_wireframe(x_e, y_e, z_e, color='lightblue', alpha=0.15)

        ax.plot(positions[:,0], positions[:,1], positions[:,2],
                color='cyan', lw=1.2, alpha=0.4, zorder=3)

        sat, = ax.plot([positions[0,0]], [positions[0,1]], [positions[0,2]],
                       'o', color='lime', ms=10,
                       markeredgecolor='white', mew=1.5, zorder=10)

        max_r = np.max(np.linalg.norm(positions, axis=1)) * 1.15
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r, max_r)

        ax.set_title(
            f"a = {a_km} km  |  e = {e}  |  i = {i_deg}°\n"
            f"Ω = {Omega_deg}°  |  ω = {omega_deg}°  |  ν = {nu_deg}°  |  {orbits:.1f} orbits",
            color='white', fontsize=10
        )
        ax.tick_params(colors='lightgray')
        fig.patch.set_facecolor('black')
        ax.grid(False)

        def update(frame):
            sat.set_data_3d([positions[frame,0]], [positions[frame,1]], [positions[frame,2]])
            return sat,

        ani = animation.FuncAnimation(fig, update, frames=len(t_eval),
                                      interval=30, blit=False, repeat=True)

        ax.view_init(elev=20, azim=30)
        plt.show()

    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as ex:
        messagebox.showerror("Error", str(ex))


# ────────────────────────────────────────────────
# GUI
# ────────────────────────────────────────────────
root = tk.Tk()
root.title("Satellite Orbit Simulator – with Presets")
root.geometry("460x580")
root.resizable(False, False)

frame = ttk.Frame(root, padding="20 20 20 20")
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Preset selector
ttk.Label(frame, text="Select Preset Orbit:").grid(column=0, row=0, sticky=tk.W, pady=8)
preset_var = tk.StringVar()
preset_combo = ttk.Combobox(frame, textvariable=preset_var,
                            values=[p[0] for p in PRESETS],
                            state="readonly", width=35)
preset_combo.grid(column=1, row=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
preset_combo.current(0)

# Input fields
entries = {}
labels = [
    "Semi-major axis (km):",
    "Eccentricity:",
    "Inclination (°):",
    "RAAN Ω (°):",
    "Arg. Perigee ω (°):",
    "True Anomaly ν (°):",
    "Orbits to show:"
]
defaults = ["6771", "0", "0", "0", "0", "0", "1.2"]

row = 1
for label, default in zip(labels, defaults):
    ttk.Label(frame, text=label).grid(column=0, row=row, sticky=tk.W, pady=4)
    entry = ttk.Entry(frame)
    entry.insert(0, default)
    entry.grid(column=1, row=row, columnspan=2, sticky=(tk.W, tk.E), pady=3)
    entries[label] = entry
    row += 1

# Load preset when selection changes
def load_preset(*args):
    selected = preset_var.get()
    for name, a, e, i, Om, om, nu in PRESETS:
        if name == selected:
            entries["Semi-major axis (km):"].delete(0, tk.END)
            entries["Semi-major axis (km):"].insert(0, str(a))
            entries["Eccentricity:"].delete(0, tk.END)
            entries["Eccentricity:"].insert(0, str(e))
            entries["Inclination (°):"].delete(0, tk.END)
            entries["Inclination (°):"].insert(0, str(i))
            entries["RAAN Ω (°):"].delete(0, tk.END)
            entries["RAAN Ω (°):"].insert(0, str(Om))
            entries["Arg. Perigee ω (°):"].delete(0, tk.END)
            entries["Arg. Perigee ω (°):"].insert(0, str(om))
            entries["True Anomaly ν (°):"].delete(0, tk.END)
            entries["True Anomaly ν (°):"].insert(0, str(nu))
            break

preset_var.trace("w", load_preset)
load_preset()  # load first preset initially

# Run button
ttk.Button(frame, text="Run Simulation",
           command=lambda: run_simulation(
               entries["Semi-major axis (km):"].get(),
               entries["Eccentricity:"].get(),
               entries["Inclination (°):"].get(),
               entries["RAAN Ω (°):"].get(),
               entries["Arg. Perigee ω (°):"].get(),
               entries["True Anomaly ν (°):"].get(),
               entries["Orbits to show:"].get()
           )).grid(column=0, row=row, columnspan=3, pady=20)

ttk.Label(frame, text="Tip: Select a preset, then click Run.\nYou can also edit values manually.",
          foreground="gray").grid(column=0, row=row+1, columnspan=3)

for child in frame.winfo_children():
    child.grid_configure(padx=6)

root.mainloop()