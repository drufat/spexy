from spexy.morphism.maps.wave import φ


def s(u, v, t):
    x, y = φ(u, v)
    z = 0
    return x, y, z
