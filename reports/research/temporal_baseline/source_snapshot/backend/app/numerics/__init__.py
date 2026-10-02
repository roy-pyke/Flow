"""Public V2 scientific-computing interface; legacy diffusion stays compatible."""
from .solver import (METHOD_BACKENDS, clear_factor_cache, factor_cache_info,
                     resolve_backend, solve, timestep_limit)
from .operators import advection_matrix, advection_rate, diffusion_matrix, diffusion_rate

__all__ = ["solve", "resolve_backend", "timestep_limit", "METHOD_BACKENDS",
           "clear_factor_cache", "factor_cache_info", "diffusion_matrix",
           "diffusion_rate", "advection_matrix", "advection_rate"]
