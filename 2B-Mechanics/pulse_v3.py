import numpy as np
from scipy.integrate import solve_ivp
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from io import BytesIO
from PIL import Image

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH  = 6371000.0        # m

PRESETS = [
    ("1. Low Earth Orbit – Equatorial", 6771, 0.00, 0.0, 0.0, 0.0, 0.0),
    ("2. Low Earth Orbit – Cape Canaveral", 6771, 0.00, 28.5, 0.0, 0.0, 0.0),
    ("3. ISS Orbit", 6791, 0.00, 51.6, 0.0, 0.0, 0.0),
    ("4. Polar Sun-Synchronous", 6971, 0.00, 98.0, 0.0, 0.0, 0.0),
    ("5. GPS Navigation Orbit", 26560, 0.00, 55.0, 0.0, 0.0, 0.0),
    ("6. Geostationary Equatorial", 42164, 0.00, 0.0, 0.0, 0.0, 0.0),
    ("7. Geostationary Transfer (GTO)", 24000, 0.73, 28.5, 0.0, 180.0, 0.0),
    ("8. Molniya Orbit", 26554, 0.72, 63.4, 0.0, 270.0, 0.0),
    ("9. Tundra Orbit", 35000, 0.27, 63.4, 0.0, 270.0, 0.0),
    ("10. Very Low Earth Orbit", 6621, 0.00, 51.6, 0.0, 0.0, 0.0),
]

# ────────────────────────────────────────────────
# Orbital mechanics
# ────────────────────────────────────────────────
def two_body_ode(t, y, mu):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.hstack((v, a))

def kepler_to_cartesian(a, e, i, Omega, omega, nu, mu):
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    def Rz(x):
        c, s = np.cos(x), np.sin(x)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Rx(x):
        c, s = np.cos(x), np.sin(x)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    Q = Rz(Omega) @ Rx(i) @ Rz(omega)
    return Q @ r_pf, Q @ v_pf

# ────────────────────────────────────────────────
# Simulation & Visualization
# ────────────────────────────────────────────────
def run_simulation(selected_name):
    try:
        preset = next(p for p in PRESETS if p[0] == selected_name)
        _, a_km, e, i_deg, Om_deg, w_deg, nu_deg = preset

        a  = a_km * 1000.0
        i  = np.deg2rad(i_deg)
        Om = np.deg2rad(Om_deg)
        w  = np.deg2rad(w_deg)
        nu = np.deg2rad(nu_deg)

        r0, v0 = kepler_to_cartesian(a, e, i, Om, w, nu, MU_EARTH)
        y0 = np.hstack((r0, v0))

        T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        t_eval = np.linspace(0, 3*T, 2000)

        sol = solve_ivp(
            two_body_ode, (0, t_eval[-1]), y0,
            args=(MU_EARTH,), rtol=1e-9, atol=1e-9, t_eval=t_eval
        )

        if not sol.success:
            raise RuntimeError("Orbit integration failed")

        positions = sol.y[:3].T / 1000.0  # km

        # ── Plotter ───────────────────────────────
        plotter = BackgroundPlotter(window_size=(1100, 900), title=selected_name)
        plotter.set_background("black")

        # ── Earth ────────────────────────────────
        earth = pv.Sphere(radius=R_EARTH/1000.0, theta_resolution=96, phi_resolution=48)
        earth.texture_map_to_sphere(inplace=True)

        texture = None
        try:
            url = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
            img = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
            texture = pv.Texture(np.array(img))
        except Exception:
            plotter.add_text("Earth texture unavailable", position="lower_right")

        plotter.add_mesh(earth, texture=texture, smooth_shading=True)

        # ── Orbit ────────────────────────────────
        orbit = pv.Spline(positions, 2000)
        plotter.add_mesh(orbit, color="cyan", line_width=3, opacity=0.7)

        # ── Satellite ────────────────────────────
        sat_actor = plotter.add_mesh(pv.Sphere(radius=80), color="#00ffcc", smooth_shading=True)

        # ── Tail ─────────────────────────────────
        tail_len = 30
        tail_poly = pv.PolyData(positions[:1])
        tail_actor = plotter.add_mesh(tail_poly, color="#00ffcc", opacity=0.8)

        plotter.add_axes()
        plotter.camera_position = "xy"
        plotter.camera.zoom(1.3)

        # ── Follow camera toggle ─────────────────
        follow_satellite = False

        def toggle_follow(flag):
            nonlocal follow_satellite
            follow_satellite = flag

        plotter.add_checkbox_button_widget(
            toggle_follow,
            value=False,
            position=(10, 10),
            size=30,
            color_on="lime",
            color_off="gray"
        )
        plotter.add_text("Follow Satellite", position=(50, 15), color="white", font_size=10)

        # ── Animation loop ───────────────────────
        # ── Animation loop ───────────────────────
        frame = 0
        follow_distance = 1500  # km behind satellite
        follow_height = 500     # km above orbital plane

        def update():
            nonlocal frame
            pos = positions[frame % len(positions)]
            sat_actor.SetPosition(pos)

            # Tail
            new_tail = np.vstack((tail_poly.points, pos))
            if len(new_tail) > tail_len:
                new_tail = new_tail[-tail_len:]
            tail_poly.points = new_tail
            tail_poly.Modified()

            # Follow camera: only adjust position relative to satellite
            if follow_satellite:
                cam_idx = (frame - 10) % len(positions)  # small lag behind satellite
                target_pos = positions[cam_idx]

                # Direction vector along orbit
                direction = pos - target_pos
                direction = direction / np.linalg.norm(direction)

                # Desired camera position (behind satellite along orbit)
                desired_cam_pos = target_pos - direction * follow_distance
                desired_cam_pos[2] += follow_height  # slightly above orbital plane

                # Smooth movement toward desired position
                alpha = 0.15
                current_pos = np.array(plotter.camera.position)
                plotter.camera.position = current_pos * (1 - alpha) + desired_cam_pos * alpha

                # Move focal point slightly toward satellite but do not force rotation
                focal_offset = (pos - np.array(plotter.camera.focal_point)) * 0.15
                plotter.camera.focal_point = np.array(plotter.camera.focal_point) + focal_offset

            frame += 1
            plotter.update()


        plotter.add_callback(update, interval=25)
        plotter.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ────────────────────────────────────────────────
# GUI
# ────────────────────────────────────────────────
root = tk.Tk()
root.title("Orbit Pulse – Two-Body Viewer")
root.geometry("420x280")
root.resizable(False, False)

frame = ttk.Frame(root, padding=30)
frame.pack(expand=True)

ttk.Label(frame, text="Select an orbit:", font=("Helvetica", 12, "bold")).pack(pady=(0, 15))

preset_var = tk.StringVar()
combo = ttk.Combobox(frame, textvariable=preset_var, values=[p[0] for p in PRESETS],
                     state="readonly", width=38)
combo.current(0)
combo.pack(pady=10)

ttk.Button(frame, text="🚀 Run Simulation", command=lambda: run_simulation(preset_var.get())).pack(pady=25)

ttk.Label(frame, text="Left-drag rotate • Wheel zoom • Follow toggle locks camera",
          foreground="gray", font=("Helvetica", 9)).pack()

root.mainloop()
