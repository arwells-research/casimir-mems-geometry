# tools/smoke_levelA.py
import numpy as np
from casimir_mems.types import Sphere, Plane, RectTrenchGrating
from casimir_mems.levelA.plane_plane import E_pp_ideal, P_pp_ideal
from casimir_mems.levelA.pfa import F_sphere_plane_pfa, dF_dd_sphere_plane_pfa
from casimir_mems.levelA.grating_pfa_mix import F_sphere_trench_pfa_mix, dF_dd_sphere_trench_pfa_mix
from casimir_mems.levelA.interface import sphere_target_curve

def main() -> None:
    d = np.array([100e-9, 200e-9, 500e-9])
    s = Sphere(R=50e-6)
    g = RectTrenchGrating(p=400e-9, w=200e-9, h=300e-9)

    E = E_pp_ideal(d)
    P = P_pp_ideal(d)
    Fsp = F_sphere_plane_pfa(d, s)
    Gsp = dF_dd_sphere_plane_pfa(d, s)
    Fsg = F_sphere_trench_pfa_mix(d, s, g)
    Gsg = dF_dd_sphere_trench_pfa_mix(d, s, g)

    assert np.all(E < 0)
    assert np.all(P < 0)
    assert np.all(Fsp < 0)
    assert np.all(Gsp < 0)
    assert np.all(Fsg < 0)
    assert np.all(Gsg < 0)

    assert np.allclose(Gsp, sphere_target_curve(d, s, Plane(), quantity="force_gradient"))
    assert np.allclose(Gsg, sphere_target_curve(d, s, g, quantity="force_gradient"))

    print("OK: Level A smoke test passed.")

if __name__ == "__main__":
    main()