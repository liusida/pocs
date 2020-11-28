# Author: Sida Liu
# This is an implementation of Hierarchic Social Entropy
# Balch, T. Hierarchic Social Entropy: An Information Theoretic Measure of Robot Group Diversity. Autonomous Robots 8, 209–238 (2000). https://doi.org/10.1023/A:1008973424594

# Main idea:
# use Numerical Taxonamy to automatically generate species, then calculate entropy summing across the swarm.

from metric import Metric


class HSEMetric(Metric):
    pass