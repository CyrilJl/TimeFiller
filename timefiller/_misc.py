# -*- coding: utf-8 -*-

# author : Cyril Joly

def check_params(param, params=None, types=None):
    """Parameter checks. Tests whether ``param`` belongs to ``params``, and checks if type(param)
    belongs to ``types``.

    This function performs tests on the parameter ``param`` to verify if it belongs to a set
    of acceptable parameters ``params`` and if its type belongs to a set of acceptable types ``types``.

    Args:
        param:
            The parameter to be tested.
        params (iterable, optional):
            The set of acceptable parameters. If specified, ``param`` must belong to this set.
            Default is None.
        types (type or iterable of types, optional):
            The set of acceptable types. If specified, the type of ``param`` must belong to this set.
            Default is None.

    Raises:
        ValueError
            If the parameter ``param`` does not satisfy the test conditions defined by ``params`` and/or ``types``.

    Returns:
        The initial parameter ``param``.

    Examples:
        .. code-block:: python

            check_params(5, params=[1, 2, 3, 4, 5])
            >>> 5
            check_params('hello', types=str)
            >>> 'hello'

    Note:
        - If ``params`` is specified, ``param`` must be an element of the ``params`` set.
        - If ``types`` is specified, the type of ``param`` must be an element of the ``types`` set.
        - If both ``params`` and ``types`` are specified, ``param`` must satisfy both conditions.
        - If ``params`` and ``types`` are set to None, no test is performed, and ``param`` is returned without modification.
    """
    if (types is not None) and (not isinstance(param, types)):
        if isinstance(types, type):
            accepted = f'{types}'
        else:
            accepted = f"{', '.join([str(t) for t in types])}"
        msg = f"`{param}` is not of an accepted type, can only be of type {accepted}!"
        raise TypeError(msg)
    if (params is not None) and (param not in params):
        msg = f"`{param}` is not a recognized argument, can only be one of {', '.join(sorted(params))}!"
        raise ValueError(msg)
    return param
