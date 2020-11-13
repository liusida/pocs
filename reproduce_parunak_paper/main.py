# Reproduce Paper: Entropy and self-organization in multi-agent systems
# https://doi.org/10.1145/375735.376024

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
fig, axes = plt.subplots(nrows=2, ncols=2)

for guide in range(2):
    total_timesteps = 250
    total_run = 30
    log_30 = np.log(total_run)
    rho = 20
    rho_2 = rho*rho

    data = np.zeros([total_run, 1, 3, total_timesteps+1])  # 30 runs, 1 agent, x-y-angle, 250 timesteps
    data[:, :, 0, 0] = data[:, :, 1, 0] = 30  # start at 30,30
    data[:, :, 2, 0] = 0  # initial angle is to the right, along x-axis

    data_molecules = np.ones([total_run, total_timesteps, 2, total_timesteps+1]) * np.inf  # 30 runs, max 250 molecules, x-y, 250 timesteps
    random_angle_molecules = np.random.random([total_run, total_timesteps, 1, total_timesteps]) * 2 * np.pi
    for timestep in range(total_timesteps):
        data_molecules[:, timestep, :, timestep] = 50
        vector_R_x = np.cos(random_angle_molecules[:, :, 0, timestep]) * 2
        vector_R_y = np.sin(random_angle_molecules[:, :, 0, timestep]) * 2
        data_molecules[:, :, 0, timestep+1] = data_molecules[:, :, 0, timestep] + vector_R_x
        data_molecules[:, :, 1, timestep+1] = data_molecules[:, :, 1, timestep] + vector_R_y

    random_angle = np.random.random([total_run, 1, 1, total_timesteps]) * 2 * np.pi
    # in paper it says steering, but it is equavalent to directly use it as angle.
    # because it's uniformly random in [0,2 pi].
    data[:, :, 2:3, 1:] = random_angle

    for timestep in range(total_timesteps):
        # T = 1
        # How could T==0 and walker wanders randomly around? I think the paper made a mistake here.
        vector_R_x = np.cos(random_angle[:, :, 0, timestep])
        vector_R_y = np.sin(random_angle[:, :, 0, timestep])
        if guide:
            # g = 1
            vector_G_x = np.zeros([total_run, 1])
            vector_G_y = np.zeros([total_run, 1])
            for i in range(total_timesteps):
                if np.isposinf(data_molecules[0, i, 0, timestep]):
                    break
                dx = data_molecules[:, i, 0, timestep].reshape([-1,1])-data[:, :, 0, timestep]
                dy = data_molecules[:, i, 1, timestep].reshape([-1,1])-data[:, :, 1, timestep]
                r_2 = (dx*dx + dy*dy)
                r_less_than_rho = r_2 < rho_2
                r_less_than_rho = r_less_than_rho.astype(int)
                one_over_r_2 = 1 / r_2
                vector_G_x += dx * one_over_r_2 * r_less_than_rho
                vector_G_y += dy * one_over_r_2 * r_less_than_rho
            vector_sum_x = vector_G_x + vector_R_x
            vector_sum_y = vector_G_y + vector_R_y
        else:
            vector_sum_x = vector_R_x
            vector_sum_y = vector_R_y
        normalize_factor = np.sqrt(vector_sum_x*vector_sum_x + vector_sum_y*vector_sum_y)
        vector_sum_x = vector_sum_x / normalize_factor
        vector_sum_y = vector_sum_y / normalize_factor
        data[:, :, 0, timestep+1] = data[:, :, 0, timestep] + vector_sum_x
        data[:, :, 1, timestep+1] = data[:, :, 1, timestep] + vector_sum_y

    ax = axes[0, guide]
    ax.set_title(f"Figure 8. with Guide={guide}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(25,55)
    ax.set_ylim(25,55)
    for i in range(total_run):
        ax.plot(data[i, 0, 0, :], data[i, 0, 1, :], linewidth=0.5)

    grid_dim_x = grid_dim_y = 100/15
    # print((99.999+1e-10) // grid_dim_x) # idx should be 0 ~ 4

    idx = (data[:,:,0,:] + 1e-10) // grid_dim_x
    idy = (data[:,:,0,:] + 1e-10) // grid_dim_y
    id_combined = idx * 1024 + idy

    macro_entropy = []
    for timestep in range(250):
        states, counts = np.unique(id_combined[:,:,timestep], return_counts=True)
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log(probabilities)) / log_30
        macro_entropy.append(entropy)

    ax = axes[1, guide]
    ax.plot(np.arange(0,250), macro_entropy)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Figure 6. with Guide={guide}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Macro Entropy")

plt.tight_layout()
plt.savefig("Fig.6 and 8.png")
plt.show()