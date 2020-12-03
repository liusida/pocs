from investigate_mi import *

if __name__ == "__main__":

    x = [1, 1, 1, 2, 2, 2, 3]
    y = [1, 2, 1, 2, 1, 2, 3]
    x = np.array(x)
    y = np.array(y)
    # investigate(x,y)

    def two_iid_rv_of_100_possible_states():
        x = np.random.randint(low=0, high=100, size=[100000])
        y = np.random.randint(low=0, high=100, size=[100000])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="100k steps")

        x = np.random.randint(low=0, high=100, size=[1000])
        y = np.random.randint(low=0, high=100, size=[1000])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="1k steps")

        x = np.random.randint(low=0, high=100, size=[10])
        y = np.random.randint(low=0, high=100, size=[10])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="10 steps")

    def two_iid_rv_of_length_33():
        x = np.random.randint(low=0, high=2, size=[33])
        y = np.random.randint(low=0, high=2, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="2 possible states")

        x = np.random.randint(low=0, high=4, size=[33])
        y = np.random.randint(low=0, high=4, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="4 possible states")

        x = np.random.randint(low=0, high=8, size=[33])
        y = np.random.randint(low=0, high=8, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x, y, title="8 possible states")

    def two_series_of_one_agent_slowly_moving_straight():
        agent_trajectory = np.linspace(start=0, stop=1, num=200)
        agent_trajectory = agent_trajectory * 100 // 10
        print(agent_trajectory)
        x = agent_trajectory[:100]
        y = agent_trajectory[-100:]
        investigate(x,y, title="One agent slowly moving straight")
    
    def one_agent_moving_faster_and_faster():
        agent_trajectory = []
        pos = 0.
        for i in range(1000):
            pos += i
            agent_trajectory.append(pos)
        agent_trajectory = np.array(agent_trajectory)
        
        for n in [10,100,1000,10000,10000]:
            agent_trajectory_bined = agent_trajectory // n
            vals = np.unique(agent_trajectory_bined)
            print(f"binned to {len(vals)} states.")
            x = agent_trajectory_bined[:100]
            y = agent_trajectory_bined[-100:]
            investigate(x,y, title=f"One agent moving faster and faster\nbinned to {len(vals)} states.")

    one_agent_moving_faster_and_faster()
