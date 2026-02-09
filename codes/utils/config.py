import argparse

class ConfigToArgsAdapter:
    """
    Adapts a configuration dictionary (e.g., from YAML) to an argparse.Namespace object.
    This allows legacy code that expects an 'args' object to work with modern YAML configs.
    """
    
    @staticmethod
    def dict_to_namespace(config_dict):
        """
        Recursively converts a dictionary to an argparse.Namespace.
        
        Args:
            config_dict (dict): The configuration dictionary.
            
        Returns:
            argparse.Namespace: The corresponding namespace object.
        """
        if not isinstance(config_dict, dict):
            return config_dict
            
        namespace = argparse.Namespace()
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursive call for nested dicts (though argparse usually is flat, 
                # some legacy code might access args.sub.param if using groups, 
                # but standard argparse is flat. We flattener or keep nested? 
                # SymUNet's option.py produces a flat namespace.
                # If YAML is structured, we might need to flatten it or the user 
                # should provide a flat YAML for compatibility.
                # STRATEGY: We convert nested dicts to nested Namespaces 
                # to support structured YAML, but legacy code likely expects flat attributes.
                # For Phase 2, we assume the YAML input structure matches what option.py would produce (mostly flat).
                setattr(namespace, key, ConfigToArgsAdapter.dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
                
        return namespace

    @staticmethod
    def apply_defaults(args, defaults):
        """
        Applies default values to the args namespace if keys are missing.
        
        Args:
            args (argparse.Namespace): The current args.
            defaults (dict): Default values.
        """
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        return args
