<<<<<<< HEAD
def check_params(param, params=None, types=None):
    """Tests d'un paramètre. Teste si ``param`` appartient à ``params``, teste si type(param)
    appartient à ``types``.

    Cette fonction effectue des tests sur le paramètre ``param`` pour vérifier s'il appartient à un ensemble
    de paramètres acceptables ``params`` et s'il a un type appartenant à un ensemble de types acceptables ``types``.

    Args:
        param:
            Le paramètre à tester.
        params (iterable, optional):
            L'ensemble des paramètres acceptables. Si spécifié, ``param`` doit appartenir à cet ensemble.
            Par défaut, None.
        types (type ou iterable de types, optional):
            L'ensemble des types acceptables. Si spécifié, le type de ``param`` doit appartenir à cet ensemble.
            Par défaut, None.

    Raises:
        ValueError
            Si le paramètre ``param`` ne satisfait pas les conditions de test définies par ``params`` et/ou ``types``.

    Returns:
        Le paramètre initial ``param``.

    Example:
=======
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
>>>>>>> d4d30e6a819def82efd804dc718545acad7fccc0
        .. code-block:: python

            check_params(5, params=[1, 2, 3, 4, 5])
            >>> 5
            check_params('hello', types=str)
            >>> 'hello'

    Note:
<<<<<<< HEAD
        - Si ``params`` est spécifié, ``param`` doit être un élément de l'ensemble ``params``.
        - Si ``types`` est spécifié, le type de ``param`` doit être un élément de l'ensemble ``types``.
        - Si les deux ``params`` et ``types`` sont spécifiés, ``param`` doit satisfaire les deux conditions.
        - Si ``params`` et ``types`` sont à None, aucun test n'est effectué, et ``param`` est retourné sans modification.
=======
        - If ``params`` is specified, ``param`` must be an element of the ``params`` set.
        - If ``types`` is specified, the type of ``param`` must be an element of the ``types`` set.
        - If both ``params`` and ``types`` are specified, ``param`` must satisfy both conditions.
        - If ``params`` and ``types`` are set to None, no test is performed, and ``param`` is returned without modification.
>>>>>>> d4d30e6a819def82efd804dc718545acad7fccc0
    """
    if (types is not None) and (not isinstance(param, types)):
        if isinstance(types, type):
            accepted = f'{types}'
        else:
            accepted = f"{', '.join([str(t) for t in types])}"
<<<<<<< HEAD
        msg = f"`{param}` n'est pas d'un type accepté, peut uniquement être du type {accepted} !"
        raise TypeError(msg)
    if (params is not None) and (param not in params):
        msg = f"`{param}` n'est pas un argument reconnu, peut uniquement valoir {', '.join(sorted(params))} !"
=======
        msg = f"`{param}` is not of an accepted type, can only be of type {accepted}!"
        raise TypeError(msg)
    if (params is not None) and (param not in params):
        msg = f"`{param}` is not a recognized argument, can only be one of {', '.join(sorted(params))}!"
>>>>>>> d4d30e6a819def82efd804dc718545acad7fccc0
        raise ValueError(msg)
    return param
