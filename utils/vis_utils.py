import numpy as np
import matplotlib.pyplot as plt


def draw_circle(x, y, r, color="k", alpha=1.0, linewidth=1):
    """ Draw a circle with center (x, y) and radius r """
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x + r * np.cos(theta), y + r * np.sin(theta), color, alpha=alpha, linewidth=linewidth)


def draw_quadrotor(x, color="k", alpha=1.0, exaggerate_theta=3, thickness=2):
    """ Draw a quadrotor with state x = (x, y, v_x, v_y, theta, omega) """
    x, y, v_x, v_y, theta, omega = x
    theta = theta * exaggerate_theta

    length = 0.06
    radius = 0.01
    plt.plot(
        [x - length * np.cos(theta), x + length * np.cos(theta)],
        [y - length * np.sin(theta), y + length * np.sin(theta)],
        color=color,
        alpha=alpha,
        linewidth=thickness,
    )
    # Draw the propellers
    for sgn in [-1, 1]:
        draw_circle(
            x + sgn * length * np.cos(theta) + 0.5 * radius * np.sin(theta),
            y + sgn * length * np.sin(theta) + 0.5 * radius * np.cos(theta),
            radius,
            color=color,
            alpha=alpha,
            linewidth=thickness,
        )


def visualize_xtraj(xtraj, interval=10, xlim=None, ylim=None):
    plt.figure()
    plt.axis("equal")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    for t in range(0, xtraj.shape[0], interval):
        draw_quadrotor(xtraj[t], alpha=(t / xtraj.shape[0]))
    plt.show()


def make_gif_from_xtraj(xtraj, filename):
    import imageio
    images = []
    fig, ax = plt.subplots()
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([0, 3.5])
    ax.axis("off")

    for t in range(0, xtraj.shape[0], 10):
        ax.clear()  # Clear the previous frame content
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([0, 3.5])

        draw_quadrotor(xtraj[t, :6])

        # Save the current frame to the images list
        fig.savefig("temp.png")
        images.append(imageio.imread("temp.png"))

    plt.close(fig)
    imageio.mimsave(filename, images, fps=10, loop=0)


def make_mp4_from_xtraj(xtraj, filename):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([0, 3.5])
    ax.axis("off")

    def update(t):
        ax.clear()
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([0, 3.5])

        draw_quadrotor(xtraj[t, :6])

    ani = animation.FuncAnimation(fig, update, frames=xtraj.shape[0], interval=100)
    ani.save(filename, writer="ffmpeg", fps=10)
