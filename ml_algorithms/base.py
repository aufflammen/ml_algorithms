from numbers import Number

class BaseModel:
    
    def __str__(self) -> str:
        params = (f'{k}={v}' for k, v in self.__dict__.items() if isinstance(v, (str, Number)))
        params_print = ', '.join(params)
        return f'{self.__class__.__name__}({params_print})'