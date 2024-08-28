import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

# Parameters
L = 10  # Length of x-axis
T = 50  # Total time
Nx = 200  # Number of points in x
Nt = 500  # Number of time steps
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
m = 1  # Mass parameter

# Initial conditions (Gaussian pulse)
sigma = 0.5
phi0 = np.exp(-(x - L/2)**2 / (2*sigma**2))
dphi0 = np.zeros_like(x)

# Compute solution
dx = x[1] - x[0]
dt = t[1] - t[0]
r = dt / dx
k = np.fft.fftfreq(Nx) * 2 * np.pi / dx
omega = np.sqrt(k**2 + m**2)

phi_k = np.fft.fft(phi0)
dphi_k = np.fft.fft(dphi0)

def update(frame):
    phi = np.real(np.fft.ifft(phi_k * np.cos(omega * t[frame]) + 
                              dphi_k / omega * np.sin(omega * t[frame])))
    line.set_ydata(phi)
    return line,

# Create the plot
fig, ax = plt.subplots(figsize=(8,6), dpi=150)
line, = ax.plot(x, phi0)
ax.set_ylim(-1, 1)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Field amplitude (φ)')
ax.set_title('Klein-Gordon Equation: Field Evolution')

# Create the animation
anim = FuncAnimation(fig, update, frames=Nt, interval=25, blit=True)

# Save the animation
writer = writers['ffmpeg'](fps=30, metadata=dict(artist='Axect'), bitrate=1800)
anim.save('kge.mp4', writer=writer)

plt.show()
