from spexy.morphism.maps.rot2 import φ


def s(u, v, t):
    x, y = φ(u, v)
    z = 0
    return x, y, z
