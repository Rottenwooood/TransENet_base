import sys
import inspect

class Registry:
    """
    A simple registry to map strings to classes or functions.
    
    Usage:
        MODELS = Registry("models")
        
        @MODELS.register()
        class MyModel:
            pass
            
        model = MODELS.get("MyModel")()
    """
    
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registered module by key."""
        return self._module_dict.get(key, None)

    def register(self, name=None):
        """
        Decorator to register a class or function.
        
        Args:
            name (str, optional): The key to register the object under.
                                  If None, uses the object's __name__.
        """
        def _register(obj):
            key = name if name is not None else obj.__name__
            if key in self._module_dict:
                print(f"WARNING: {key} is already registered in {self._name}. Overwriting.")
            self._module_dict[key] = obj
            return obj
        return _register
    
    def build(self, key, **kwargs):
        """
        Instantiate a registered class or call a function.
        
        Args:
            key (str): The key of the registered object.
            **kwargs: Arguments passed to the constructor/function.
            
        Returns:
            The instance or return value.
            
        Raises:
            KeyError: If key is not registered.
        """
        cls = self.get(key)
        if cls is None:
            raise KeyError(f"'{key}' is not registered in {self._name}. Available: {list(self._module_dict.keys())}")
        return cls(**kwargs)

# Global Registries
ARCH_REGISTRY = Registry("ARCH")
LOSS_REGISTRY = Registry("LOSS")
