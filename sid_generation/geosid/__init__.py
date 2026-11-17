"""Optional, quantizer-aware geometry shaping; backends are loaded on demand."""

BACKENDS = ("rqvae", "rq-opq")


def backend_class(name):
    if name == "rqvae":
        from .rqvae_backend import RQVAEBackend
        return RQVAEBackend
    if name == "rq-opq":
        from .adapters.rq_opq import RQOPQBackend
        return RQOPQBackend
    raise ValueError(f"unsupported GeoSID backend {name!r}; choose from {BACKENDS}")
