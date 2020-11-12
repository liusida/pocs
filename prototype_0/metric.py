class Metric:
    def __init__(self, world):
        self.world = world
        
    def get_metric(self):
        return self.world.vehicles[0].pos_x / self.world.width  # TODO: for demostration. Should be entropy or something.
