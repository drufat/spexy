from importlib import import_module


def get_map(name):
    """
    >>> get_map('identity')
    Map((s, t), (s, t))
    """
    return import_module('spexy.morphism.maps.{}'.format(name)).φ
