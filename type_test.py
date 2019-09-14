# """
from typing import Callable, Union


def with_func_attrs(**attrs: Union[int, str, ]) -> Callable:
    ''' with_func_attrs '''
    def with_attrs(fct):
        for key, val in attrs.items():
            setattr(fct, key, val)
        return fct
    return with_attrs


@with_func_attrs(attr=-1)
# """
def func() -> None:
    func.attr = 0
    # pass
