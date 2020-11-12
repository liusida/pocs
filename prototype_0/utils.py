from timeit import default_timer as timer

def log(*args):
    if True:
        print(*args)

g_start = 0
def reset_timer(description=None):
    global g_start
    if description is not None:
        end = timer()
        print(f"reset_timer: {description} {end-g_start:.03f}")
    g_start = timer()