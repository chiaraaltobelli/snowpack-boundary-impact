# tests/conftest.py
import sys
import types

from pathlib import Path
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
def _ensure_stub(fullname, attrs: dict):
    """Ensure that a stub module with given attributes exists in sys.modules."""
    if fullname in sys.modules:
        return
    mod = types.ModuleType(fullname)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[fullname] = mod

try:
    import cartopy
except Exception:
    # stub cartopy.crs and cartopy.feature so imports succeed
    class _CRS: # no-op placeholder
        class PlateCarree:
            def __init__(self, *a, **k): 
                pass
    class _Feature:
        def with_scale(self, *a, **k): 
            return self
    # register stub modules
    _ensure_stub("cartopy", {})
    _ensure_stub("cartopy.crs", {"PlateCarree"  : _CRS.PlateCarree})
    _ensure_stub("cartopy.feature", {
        "COASTLINE": _Feature(),
        "BORDERS": _Feature(),
        "STATES": _Feature(),
        "RIVERS": _Feature(),
        "LAKES": _Feature()
    })