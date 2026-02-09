import sys
import pytest
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import Registry, ARCH_REGISTRY, LOSS_REGISTRY

def test_registry_basic():
    """Test basic registration and retrieval"""
    reg = Registry("TEST")
    
    @reg.register()
    class Foo:
        pass
        
    assert "Foo" in reg
    assert reg.get("Foo") == Foo

def test_registry_custom_name():
    """Test registration with custom name"""
    reg = Registry("TEST")
    
    @reg.register("BarModel")
    class Bar:
        pass
        
    assert "BarModel" in reg
    assert reg.get("BarModel") == Bar

def test_registry_build():
    """Test building objects"""
    reg = Registry("TEST")
    
    @reg.register()
    class Configurable:
        def __init__(self, a, b=1):
            self.a = a
            self.b = b
            
    obj = reg.build("Configurable", a=10, b=20)
    assert obj.a == 10
    assert obj.b == 20
    
    with pytest.raises(KeyError):
        reg.build("NonExistent")

def test_global_registries():
    """Test that global registries exist and function"""
    assert ARCH_REGISTRY.name == "ARCH"
    assert LOSS_REGISTRY.name == "LOSS"

if __name__ == "__main__":
    from unittest.mock import MagicMock
    # Patch sys.argv for pytest call within script if needed
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
