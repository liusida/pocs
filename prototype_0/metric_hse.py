# Author: Sida Liu
# This is an implementation of Hierarchic Social Entropy
# Balch, T. Hierarchic Social Entropy: An Information Theoretic Measure of Robot Group Diversity. Autonomous Robots 8, 209–238 (2000). https://doi.org/10.1023/A:1008973424594

# Main idea:
# use Numerical Taxonamy to automatically generate species, then calculate entropy summing across the swarm.

# Social Entropy was proposed in (Bailey, 1990)
# Balch adopts the terminology to robotic swarm with slightly different definition.
# Simple Social Entropy in Balch's work is just Shannon's Entropy with specific definition of state and individual.
# Limitation: can't tell how differet the states are. e.g. a rat and a mouse, v.s. a worm and a elephant.

# Numerical Taxonomy is a biology field that orgnizing individuals into a tree accroding to their classification position.
import scipy.stats
import numpy as np
from metric import Metric


class HSEMetric(Metric):
    def distance(self, point1, point2):
        d = np.abs(point1 - point2)
        if d[0]>0.5:
            d[0] = 1-d[0]
        if d[1]>0.5:
            d[1] = 1-d[1]
        return np.sqrt(d[0]*d[0] + d[1]*d[1])
        # return np.linalg.norm(point1 - point2)

    def clustering(self, data, h):
        """Implementation of Cu algorithm for clustering at level h
        assuming data is a 2D numpy array, contains the classification position of each agent.
        h is a number between 0 and 1. 0: every individual is a cluster. 1: all individuals are in one cluster.
        """
        dimension = data.shape[1]
        h = h * np.sqrt(dimension) # h need to be normalized to 0 ~ max possible distance
        num_agents = data.shape[0]
        clusters = []
        for i in range(num_agents):
            clusters.append({i})
        
        for i, c in enumerate(clusters):
            for j in range(num_agents):
                if j==i:
                    continue
                for k in c:
                    if self.distance(data[k], data[j]) > h:
                        continue
                    c.add(j)
                    break
        
        # Discard redundant clusters
        unique_clusters = {}
        for c in clusters:
            c = tuple(c)
            if c in unique_clusters:
                unique_clusters[c] += 1
            else:
                unique_clusters[c] = 1
        unique_clusters = list(unique_clusters.keys())

        return unique_clusters

    def social_entropy(self, unique_clusters):
        total = 0
        pk = []
        for c in unique_clusters:
            total += len(c)
        for c in unique_clusters:
            pk.append(len(c)/total)
        return scipy.stats.entropy(pk, base=2)

    def get_metric(self):
        # integrate h from 0.0 to 1.0
        HSE = 0
        num_hs = 20 # discretize h
        hs = np.linspace(0., 1., num=num_hs)
        agents = self.world.current_obs[0]
        agents = agents.reshape([self.world.num_vehicles,4])
        agents = agents[:,:2]
        for h in hs:
            clusters = self.clustering(agents, h)
            # print(clusters)
            entropy = self.social_entropy(clusters)
            HSE += entropy
        HSE /= num_hs
        # print(HSE)
        return {"HSE":HSE}

if __name__ == "__main__":
    """Testing"""
    agents = np.random.random(size=[3,2]) # three agents, two classification dimensions
    print(f"check: log_2(3) = {np.log2(3)}")
    h = HSEMetric(None)
    for i in range(11):
        ret = h.clustering(agents, i*0.1)
        print(ret)
        ret = h.social_entropy(ret)
        print(ret)
