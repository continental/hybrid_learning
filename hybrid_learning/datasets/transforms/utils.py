"""Simple utility functions common to different types of transformations,
and trafo builders.
E.g. representation generation.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

from typing import Dict


def settings_to_repr(obj, settings: Dict) -> str:
    """Given an object and a dict of its settings, return a representation str.
    The object is just used to derive the class name.

    :meta private:
    """
    return "{}({})".format(str(obj.__class__.__name__),
                           ', '.join(['='.join([str(k), repr(v)])
                                      for k, v in settings.items()]))
