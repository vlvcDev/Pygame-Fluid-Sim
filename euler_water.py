#Put moveable object in the sim
import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()

# Grid parameters
nx, ny = 64, 64  # Grid size
cell_size = 8   # Size of each cell in pixels

# Simulation parameters
dt = 0.1
iterations = 30  # Number of iterations for the pressure solver
viscosity = 0.4  # Reduced viscosity coefficient

# Screen dimensions
width, height = nx * cell_size, ny * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Eulerian Water Simulation")

# Colors
background_color = (0, 0, 0)  # Black

# Initialize velocity fields with zeros
u = np.zeros((nx, ny))  # Horizontal velocity
v = np.zeros((nx, ny))  # Vertical velocity

# Initialize pressure field
pressure = np.zeros((nx, ny))

def advect(u, v, dt):
    u_new = np.copy(u)
    v_new = np.copy(v)
    for i in range(1, nx - 2):
        for j in range(1, ny - 2):
            # Compute previous positions
            x_prev = i - u[i, j] * dt / cell_size
            y_prev = j - v[i, j] * dt / cell_size
            # Bound the positions within the grid
            x_prev = min(max(x_prev, 1), nx - 3)
            y_prev = min(max(y_prev, 1), ny - 3)
            # Get integer parts
            i0, j0 = int(x_prev), int(y_prev)
            # Linear interpolation weights
            s1, t1 = x_prev - i0, y_prev - j0
            s0, t0 = 1 - s1, 1 - t1
            # Interpolate velocities
            u_new[i, j] = (
                s0 * (t0 * u[i0, j0] + t1 * u[i0, j0 + 1]) +
                s1 * (t0 * u[i0 + 1, j0] + t1 * u[i0 + 1, j0 + 1])
            )
            v_new[i, j] = (
                s0 * (t0 * v[i0, j0] + t1 * v[i0, j0 + 1]) +
                s1 * (t0 * v[i0 + 1, j0] + t1 * v[i0 + 1, j0 + 1])
            )
    enforce_boundaries(u_new, v_new)
    return u_new, v_new

def diffuse(u, v, viscosity, dt):
    u_new = np.copy(u)
    v_new = np.copy(v)
    a = dt * viscosity * (nx - 2) * (ny - 2)
    for _ in range(iterations):
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1] + a * (
                u_new[2:, 1:-1] + u_new[:-2, 1:-1] + u_new[1:-1, 2:] + u_new[1:-1, :-2]
            )
        ) / (1 + 4 * a)
        v_new[1:-1, 1:-1] = (
            v[1:-1, 1:-1] + a * (
                v_new[2:, 1:-1] + v_new[:-2, 1:-1] + v_new[1:-1, 2:] + v_new[1:-1, :-2]
            )
        ) / (1 + 4 * a)
        enforce_boundaries(u_new, v_new)
    return u_new, v_new

def compute_divergence(u, v):
    divergence = np.zeros((nx, ny))
    divergence[1:-1, 1:-1] = (
        (u[2:, 1:-1] - u[:-2, 1:-1]) +
        (v[1:-1, 2:] - v[1:-1, :-2])
    ) / (-2 * cell_size)
    return divergence

def pressure_projection(u, v, pressure, divergence):
    p = np.zeros_like(pressure)
    for _ in range(iterations):
        p[1:-1, 1:-1] = (
            divergence[1:-1, 1:-1] +
            p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]
        ) / 4
        enforce_pressure_boundaries(p)
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / cell_size
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / cell_size
    enforce_boundaries(u, v)
    return u, v, p

def enforce_boundaries(u, v):
    u[:, 0] = 0  # Zero horizontal velocity at top boundary
    u[:, -1] = 0  # Zero horizontal velocity at bottom boundary

    v[0, :] = 0  # Zero vertical velocity at left boundary
    v[-1, :] = 0  # Zero vertical velocity at right boundary

    u[0, :] = -u[1, :]   # Left boundary
    u[-1, :] = -u[-2, :] # Right boundary

    v[:, 0] = -v[:, 1]   # Top boundary
    v[:, -1] = -v[:, -2] # Bottom boundary


def enforce_pressure_boundaries(p):
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]

def map_pressure_to_color(pressure):
    min_p = -0.05
    max_p = 0.05
    pressure_clipped = np.clip(pressure, min_p, max_p)
    normalized_p = (pressure_clipped - min_p) / (max_p - min_p)
    colors = np.zeros((nx, ny, 3), dtype=np.uint8)
    colors[..., 2] = (normalized_p * 255).astype(np.uint8)
    colors[..., 0] = (255 - normalized_p * 255).astype(np.uint8)
    return colors

def visualize(screen, u, v, colors):
    # Draw the pressure field as colored rectangles
    for i in range(nx):
        for j in range(ny):
            # Define the rectangle for each grid cell
            rect = pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
            # Get the color for the current grid cell
            color = tuple(colors[i, j])
            # Draw the rectangle with the corresponding color
            pygame.draw.rect(screen, color, rect)

    # Overlay the velocity vectors (arrows)
    for i in range(0, nx, 2):  # Skip some grid points for clarity
        for j in range(0, ny, 2):
            # Compute the position on the screen
            x = i * cell_size + cell_size // 2
            y = j * cell_size + cell_size // 2
            # Scale velocity for visualization
            scale = 5.0
            end_x = x + int(u[i, j] * scale)
            end_y = y + int(v[i, j] * scale)
            # Draw the velocity vector (arrow)
            pygame.draw.line(screen, (0, 0, 0), (x, y), (end_x, end_y), 1)

    pygame.display.flip()

# Initialize previous mouse position
prev_mouse_x, prev_mouse_y = pygame.mouse.get_pos()

# Main simulation loop
running = True
clock = pygame.time.Clock()
while running:
    dt = clock.tick(60) / 1000.0

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current mouse position
        # Get current mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_dx = mouse_x - prev_mouse_x
    mouse_dy = mouse_y - prev_mouse_y
    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

    # --- Mouse Interaction ---
    i = int(mouse_x / cell_size)
    j = int(mouse_y / cell_size)
    if 2 <= i < nx - 2 and 2 <= j < ny - 2:
        force_strength = 4  # Increased force strength
        u[i-2:i+3, j-2:j+3] += force_strength * mouse_dx / cell_size
        v[i-2:i+3, j-2:j+3] += force_strength * mouse_dy / cell_size

    # --- Mouse Interaction: Radial Force when Mouse is Pressed ---
    mouse_pressed = pygame.mouse.get_pressed()

    if mouse_pressed[0]:  # Left mouse button pressed
        i = int(mouse_x / cell_size)
        j = int(mouse_y / cell_size)
        if 2 <= i < nx - 2 and 2 <= j < ny - 2:
            force_strength = 10  # Adjust force strength as needed
            radius = 3  # Radius of influence
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    ni, nj = i + di, j + dj
                    # Check bounds to avoid out-of-bounds errors
                    if 0 <= ni < nx and 0 <= nj < ny:
                        dist = np.sqrt(di**2 + dj**2)
                        if dist <= radius and dist > 0:
                            direction_u = di / dist  # Horizontal direction
                            direction_v = dj / dist  # Vertical direction
                            u[ni, nj] += force_strength * direction_u / dist
                            v[ni, nj] += force_strength * direction_v / dist

    # Simulation steps
    u, v = advect(u, v, dt)
    u, v = diffuse(u, v, viscosity, dt)
    divergence = compute_divergence(u, v)
    u, v, pressure = pressure_projection(u, v, pressure, divergence)
    colors = map_pressure_to_color(pressure)
    visualize(screen, u, v, colors)

pygame.quit()
sys.exit()
