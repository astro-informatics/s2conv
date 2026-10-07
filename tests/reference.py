"""
Independent NumPy references for the s2conv tests.

Nothing in this module imports s2conv.  The references follow the definitions of the
article "Unifying spherical convolutions through lifting" (J. D. McEwen) literally, and
by a different route from the package wherever there is one:

- Wigner functions come from s2fft's Risbo recursion, evaluated at every angle needed,
  rather than from the factorisation of Eq. (B25) that s2conv uses;
- spin fields are evaluated from the definition of the spin spherical harmonics (Eq. A7),
  and functions on the rotation group from the inverse Wigner transform (Eq. 7);
- rotations act in real space with the spin twist phase (Eq. 12);
- transforms are s2fft's NumPy transforms (s2conv's default backend is s2fft's JAX one);
- kernels are held as blocks ``block[el, L - 1 + n, L - 1 + k]`` = Psi^l_{nk} (first
  index n, second index k), converted from and to the s2fft layout of s2conv,
  ``kernel[N - 1 + k, el, L - 1 + n]``, by explicit loops.
"""

import numpy as np
import s2fft
from s2fft.recursions import risbo
from s2fft.sampling import s2_samples

FOUR_PI = 4 * np.pi
EIGHT_PI2 = 8 * np.pi**2
SAMPLING = "mw"
TOL = 1e-10  # relative error of an exact identity

# ------------------------------------------------------------------------------ indexing


def m_axis(L):
    """Orders -(L-1), ..., L-1 along the last axis of a coefficient array."""
    return np.arange(-(L - 1), L)


def n_axis(N):
    """Orders -(N-1), ..., N-1 along the first axis of Wigner coefficients."""
    return np.arange(-(N - 1), N)


def parity(s):
    """(-1)^s for any integer, negative ones included."""
    return 1 - 2 * (int(s) % 2)


def lm_mask(L, s=0):
    """Entries (el, m) carried by a spin-s field: el >= |s| and |m| <= el."""
    el = np.arange(L)[:, None]
    m = m_axis(L)[None, :]
    return (el >= abs(s)) & (np.abs(m) <= el)


def n_of(flmn):
    """Azimuthal band-limit N of Wigner coefficients flmn[..., N - 1 + n, el, L - 1 + m]."""
    return (flmn.shape[-3] + 1) // 2


def rel_err(a, b):
    """Maximum absolute difference relative to the maximum absolute value of the reference b."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def inner_s2(flm, glm):
    """<f, g> on the sphere by Parseval: sum_lm f_lm g*_lm."""
    return complex(np.sum(np.asarray(flm) * np.conj(np.asarray(glm))))


def inner_so3(flmn, glmn):
    """<F, G> on the rotation group by Parseval (Eq. 7): sum_l (2l+1)/(8 pi^2) sum_mn F G*."""
    L = flmn.shape[-2]
    weight = ((2 * np.arange(L) + 1) / EIGHT_PI2)[:, None]
    return complex(np.sum(weight * np.asarray(flmn) * np.conj(np.asarray(glmn))))


# ------------------------------------------------------------------------ random signals


def random_block(L, rng, N=None):
    """A random kernel block Psi^l_{nk}, |n|, |k| <= el; with N, only |n|, |k| < N."""
    el = np.arange(L)[:, None, None]
    n, k = m_axis(L)[None, :, None], m_axis(L)[None, None, :]
    support = (np.abs(n) <= el) & (np.abs(k) <= el)
    if N is not None:
        support &= (np.abs(n) < N) & (np.abs(k) < N)
    shape = (L, 2 * L - 1, 2 * L - 1)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * support


def block_to_kernel(block, N):
    """The s2fft layout kernel[N - 1 + k, el, L - 1 + n] = Psi^l_{nk} of a block, |k| < N."""
    L = block.shape[0]
    kernel = np.zeros((2 * N - 1, L, 2 * L - 1), np.complex128)
    for k in range(-(N - 1), N):
        kernel[N - 1 + k] = block[:, :, L - 1 + k]
    return kernel


def kernel_to_block(kernel):
    """The block Psi[el, L - 1 + n, L - 1 + k] of a kernel in the s2fft layout."""
    kernel = np.asarray(kernel)
    N, L = n_of(kernel), kernel.shape[-2]
    block = np.zeros((L, 2 * L - 1, 2 * L - 1), np.complex128)
    for k in range(-(N - 1), N):
        block[:, :, L - 1 + k] = kernel[N - 1 + k]
    return block


# ---------------------------------------------------------------------------- transforms


def sht(f, L, s=0):
    """Forward spin-s spherical harmonic transform of MW samples (s2fft, NumPy)."""
    f = np.asarray(f, np.complex128)
    return np.asarray(s2fft.forward(f, L, spin=s, sampling=SAMPLING, method="numpy"))


def isht(flm, L, s=0):
    """Inverse spin-s spherical harmonic transform to MW samples (s2fft, NumPy)."""
    flm = np.asarray(flm, np.complex128)
    return np.asarray(s2fft.inverse(flm, L, spin=s, sampling=SAMPLING, method="numpy"))


def wigner_fwd(F, L, N):
    """Forward Wigner transform (Eq. 7) of MW samples F[j, t, p] (s2fft, NumPy)."""
    F = np.asarray(F, np.complex128)
    return np.asarray(s2fft.wigner.forward(F, L, N, sampling=SAMPLING, method="numpy"))


def wigner_inv(flmn, L, N):
    """Inverse Wigner transform (Eq. 7) to MW samples F[j, t, p] (s2fft, NumPy)."""
    flmn = np.asarray(flmn, np.complex128)
    return np.asarray(
        s2fft.wigner.inverse(flmn, L, N, sampling=SAMPLING, method="numpy")
    )


def sphere_grid(L):
    """Colatitudes and longitudes of the MW sampling of the sphere."""
    return s2_samples.thetas(L, SAMPLING), s2_samples.phis_equiang(L, SAMPLING)


def gamma_grid(N):
    """The 2N - 1 samples of the third Euler angle, gamma_j = 2 pi j / (2N - 1)."""
    return 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)


# ------------------------------------------------------------------------ Wigner functions


def wigner_d(beta, L):
    """d^l_{mn}(beta) for el < L at a scalar angle, [L, 2L - 1, 2L - 1], by the Risbo recursion."""
    d = np.zeros((L, 2 * L - 1, 2 * L - 1))
    plane = np.zeros((2 * L - 1, 2 * L - 1))
    for el in range(L):
        plane = risbo.compute_full(plane, float(beta), L, el)
        d[el] = plane
    return d


def wigner_D(alpha, beta, gamma, L):  # noqa: N802
    """D^l_{mn}(alpha, beta, gamma) = e^{-im alpha} d^l_{mn}(beta) e^{-in gamma} (Eq. 6)."""
    m = m_axis(L)
    phase_a = np.exp(-1j * m * alpha)[None, :, None]
    phase_g = np.exp(-1j * m * gamma)[None, None, :]
    return phase_a * wigner_d(beta, L) * phase_g


def _risbo_planes(beta, L):
    """Yield (el, d) with d[p, L - 1 + m, L - 1 + n] = d^l_{mn}(beta_p), el = 0, ..., L - 1."""
    dl = np.zeros((beta.size, 2 * L - 1, 2 * L - 1))
    for el in range(L):
        dl = risbo.compute_full_vectorised(dl, beta, L, el)
        yield el, dl


def eval_spin(flm, s, theta, phi):
    """A band-limited spin-s field at arbitrary points, by definition (Eq. A7).

    sY_lm(theta, phi) = (-1)^s sqrt((2l+1)/(4 pi)) d^l_{m,-s}(theta) e^{im phi}, with the
    Wigner functions from the Risbo recursion at every distinct colatitude.
    Leading axes of flm are kept.
    """
    flm = np.asarray(flm)
    L = flm.shape[-2]
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    shape = theta.shape
    unique, inverse = np.unique(theta.ravel(), return_inverse=True)
    coeffs = flm * lm_mask(L, s)
    radial = np.zeros(flm.shape[:-2] + (unique.size, 2 * L - 1), np.complex128)
    for el, d in _risbo_planes(unique, L):
        if el >= abs(s):
            weight = parity(s) * np.sqrt((2 * el + 1) / FOUR_PI)
            radial += weight * coeffs[..., el, None, :] * d[:, :, L - 1 - s]
    phases = np.exp(1j * np.outer(phi.ravel(), m_axis(L)))
    values = np.sum(radial[..., inverse, :] * phases, axis=-1)
    return values.reshape(flm.shape[:-2] + shape)


def eval_so3(flmn, alpha, beta, gamma):
    """F(alpha, beta, gamma) = sum_l (2l+1)/(8 pi^2) sum_mn F^l_{mn} D^{l*}_{mn} (Eq. 7).

    At arbitrary rotations (broadcast), with the Wigner functions from the Risbo recursion.
    """
    flmn = np.asarray(flmn)
    N, L = n_of(flmn), flmn.shape[-2]
    angles = np.broadcast_arrays(*(np.asarray(x, float) for x in (alpha, beta, gamma)))
    shape = angles[0].shape
    alpha, beta, gamma = (x.ravel() for x in angles)
    unique, inverse = np.unique(beta, return_inverse=True)
    e_alpha = np.exp(1j * np.outer(alpha, m_axis(L)))
    e_gamma = np.exp(1j * np.outer(gamma, n_axis(N)))
    total = np.zeros(alpha.size, np.complex128)
    for el, d in _risbo_planes(unique, L):
        d_el = d[inverse][:, :, L - N : L + N - 1]
        weight = (2 * el + 1) / EIGHT_PI2
        total += weight * np.einsum(
            "nm,pmn,pm,pn->p", flmn[:, el, :], d_el, e_alpha, e_gamma
        )
    return total.reshape(shape)


# ------------------------------------------------------------------------------ rotations


def _rot_z(a):
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack(
        [np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2
    )


def _rot_y(b):
    c, s, z, o = np.cos(b), np.sin(b), np.zeros_like(b), np.ones_like(b)
    return np.stack(
        [np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2
    )


def rot_matrix(alpha, beta, gamma):
    """R(alpha, beta, gamma) = R_z(alpha) R_y(beta) R_z(gamma) (Eq. 1), as a product of matrices."""
    alpha, beta, gamma = np.broadcast_arrays(
        *(np.asarray(x, float) for x in (alpha, beta, gamma))
    )
    return _rot_z(alpha) @ _rot_y(beta) @ _rot_z(gamma)


def euler_zyz(R):
    """Euler angles (a, b, c) with R = R_z(a) R_y(b) R_z(c), b in [0, pi]."""
    a = np.arctan2(R[..., 1, 2], R[..., 0, 2])
    b = np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))
    c = np.arctan2(R[..., 2, 1], -R[..., 2, 0])
    return a, b, c


def section(theta, phi):
    """The coordinate section s(theta, phi) = R(phi, theta, 0) (Eq. 32)."""
    return rot_matrix(phi, theta, 0.0)


def unit_vector(theta, phi):
    """Cartesian unit vectors of the points (theta, phi)."""
    st = np.sin(theta)
    return np.stack(
        [st * np.cos(phi), st * np.sin(phi), np.cos(theta) + 0 * phi], axis=-1
    )


def angles_of(v):
    """Colatitudes and longitudes of unit vectors v[..., 3]."""
    theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    return theta, np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)


def random_rotation(rng):
    """Euler angles of a rotation drawn from the Haar measure."""
    return (
        float(rng.uniform(0, 2 * np.pi)),
        float(np.arccos(rng.uniform(-1, 1))),
        float(rng.uniform(0, 2 * np.pi)),
    )


def random_points(rng, n, z_max=0.95):
    """Random points on the sphere away from the poles, |cos theta| <= z_max."""
    return np.arccos(rng.uniform(-z_max, z_max, n)), rng.uniform(0, 2 * np.pi, n)


def rotate_flm(flm, rotation):
    """(R_rho f)_{lm} = sum_n D^l_{mn}(rho) f_{ln} (Eq. 10): the same matrices for every spin."""
    L = flm.shape[-2]
    return np.einsum("lmn,...ln->...lm", wigner_D(*rotation, L), flm)


def rotate_flmn(flmn, rotation):
    """Left translation (pi_G(rho) F)^l_{mn} = sum_k D^l_{mk}(rho) F^l_{kn}, on the first index."""
    L = flmn.shape[-2]
    return np.einsum("lmk,...nlk->...nlm", wigner_D(*rotation, L), flmn)


def rotated_values(psi_lm, s, rotation, theta, phi):
    """(R_rho psi)(omega) = e^{-is vartheta} psi(R_rho^{-1} omega) in real space (Eq. 12).

    vartheta is the third Euler angle of R_rho^{-1} s(omega).
    """
    R_inv = np.swapaxes(rot_matrix(*rotation), -1, -2)
    a, b, c = euler_zyz(R_inv @ section(theta, phi))
    return np.exp(-1j * s * c) * eval_spin(psi_lm, s, b, a)


# ----------------------------------------------------------------------------- quadrature


def gl_quadrature(L):
    """Gauss-Legendre nodes in cos(theta) and uniform longitudes on the sphere.

    Exact for fields band-limited at 2L + 1, hence for products of two fields band-limited at L.
    """
    x, w = np.polynomial.legendre.leggauss(L + 1)
    n_phi = 2 * L + 1
    T, P = np.meshgrid(
        np.arccos(x), 2 * np.pi * np.arange(n_phi) / n_phi, indexing="ij"
    )
    return T, P, w[:, None] * (2 * np.pi / n_phi) * np.ones_like(T)


def so3_quadrature(L):
    """Gauss-Legendre in cos(beta), uniform in alpha and gamma: exact on SO(3) up to degree 2L + 1."""
    x, w = np.polynomial.legendre.leggauss(L + 1)
    n = 2 * L + 1
    ang = 2 * np.pi * np.arange(n) / n
    A, B, G = np.meshgrid(ang, np.arccos(x), ang, indexing="ij")
    return A, B, G, (2 * np.pi / n) ** 2 * w[None, :, None] * np.ones_like(A)


def inner_with_rotated(f_lm, psi_lm, s, rotation, bilinear=False):
    """<f, R_rho psi> = int f (R_rho psi)^* d omega by exact quadrature, the kernel rotated in real space.

    Definitions III.1, III.3, III.5-III.7; bilinear=True drops the conjugation (Definition III.8).
    """
    T, P, W = gl_quadrature(f_lm.shape[-2])
    k = rotated_values(psi_lm, s, rotation, T, P)
    return complex(
        np.sum(W * eval_spin(f_lm, s, T, P) * (k if bilinear else np.conj(k)))
    )


def left_convolution_quadrature(f_lm, h_lm, theta, phi):
    """(f *_L h)(omega) = int_SO(3) f(rho eta) h(R_rho^{-1} omega) d mu(rho) (Definition III.2), by quadrature."""
    A, B, G, W = so3_quadrature(f_lm.shape[-2])
    R_inv = np.swapaxes(rot_matrix(A, B, G), -1, -2)
    f_rho_eta = eval_spin(f_lm, 0, B, A)
    out = []
    for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi), strict=True):
        tt, pp = angles_of(R_inv @ unit_vector(t, p))
        out.append(np.sum(W * f_rho_eta * eval_spin(h_lm, 0, tt, pp)))
    return np.array(out)


def synthesis_quadrature(w_lmn, psi_lm, s, theta, phi):
    """(W (*)^dagger psi)(omega) = int_SO(3) W(rho) (R_rho psi)(omega) d mu(rho) (Definition III.10), by quadrature."""
    A, B, G, W = so3_quadrature(w_lmn.shape[-2])
    w_values = eval_so3(w_lmn, A, B, G)
    R_inv = np.swapaxes(rot_matrix(A, B, G), -1, -2)
    out = []
    for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi), strict=True):
        a, b, c = euler_zyz(R_inv @ section(t, p))
        kernel = np.exp(-1j * s * c) * eval_spin(psi_lm, s, b, a)
        out.append(np.sum(W * w_values * kernel))
    return np.array(out)


def axisymmetric_synthesis_quadrature(w_lm, psi_lm, s, theta, phi):
    """int_S2 W(omega') (R_{omega'} psi)(omega) d omega', R_{omega'} = R(phi', theta', 0), by quadrature (Eq. 26)."""
    T, P, W = gl_quadrature(w_lm.shape[-2])
    w_values = eval_spin(w_lm, 0, T, P)
    R_inv = np.swapaxes(rot_matrix(P, T, 0.0), -1, -2)
    out = []
    for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi), strict=True):
        a, b, c = euler_zyz(R_inv @ section(t, p))
        out.append(
            np.sum(W * w_values * np.exp(-1j * s * c) * eval_spin(psi_lm, s, b, a))
        )
    return np.array(out)


def radial_profile(c, s, sp, b):
    """K(b) = sum_l (2l+1)/(4 pi) c_l d^l_{s's}(b), the radial part of a spin s -> s' multiplier (Cor. V.7)."""
    b = np.asarray(b, float)
    unique, inverse = np.unique(b.ravel(), return_inverse=True)
    L = len(c)
    out = np.zeros(unique.size, np.complex128)
    for el, d in _risbo_planes(unique, L):
        out += (2 * el + 1) / FOUR_PI * c[el] * d[:, L - 1 + sp, L - 1 + s]
    return out[inverse].reshape(b.shape)


def two_point_kernel(c, s, sp, x_theta, x_phi, y_theta, y_phi, literal_angles=False):
    """K(x, y) = e^{is'a} K(b) e^{isc}, s(x)^{-1} s(y) = R_z(a) R_y(b) R_z(c) (Cor. V.7, Eq. A10).

    literal_angles=True uses the printed first angle a + pi of Tessore et al.
    """
    rel = np.swapaxes(section(x_theta, x_phi), -1, -2) @ section(y_theta, y_phi)
    a, b, cc = euler_zyz(rel)
    if literal_angles:
        a = a + np.pi
    return np.exp(1j * sp * a) * radial_profile(c, s, sp, b) * np.exp(1j * s * cc)


def two_point_quadrature(c, s, sp, f_lm, theta, phi, literal_angles=False):
    """(T f)(x) = int K(x, y) f(y) d omega(y) by exact quadrature with the two-point kernel."""
    T, P, W = gl_quadrature(f_lm.shape[-2])
    fy = eval_spin(f_lm, s, T, P)
    return np.array(
        [
            np.sum(W * two_point_kernel(c, s, sp, t, p, T, P, literal_angles) * fy)
            for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi), strict=True)
        ]
    )


def needlet_window(xi, B=2.0):
    """The needlet window b(xi) of Marinucci et al.: supported on [1/B, B], sum_j b(xi/B^j)^2 = 1 for xi >= 1."""
    u = np.linspace(-1, 1, 20001)
    bump = np.zeros_like(u)
    bump[1:-1] = np.exp(-1.0 / (1.0 - u[1:-1] ** 2))
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (bump[1:] + bump[:-1]) * np.diff(u))]
    )
    cumulative /= cumulative[-1]

    def plateau(t):
        t = np.asarray(t, float)
        middle = np.interp(1 - 2 * B / (B - 1) * (t - 1 / B), u, cumulative)
        return np.where(t <= 1 / B, 1.0, np.where(t >= 1, 0.0, middle))

    return np.sqrt(np.clip(plateau(np.asarray(xi, float) / B) - plateau(xi), 0.0, None))


# --------------------------------------------------------------------- lift and projection


def lift(flm, s, N):
    """The lift in harmonic space (Lemma IV.7(c), Eq. 38): the single Wigner column n = -s."""
    L = flm.shape[-2]
    out = np.zeros((2 * N - 1, L, 2 * L - 1), np.complex128)
    weight = parity(s) * np.sqrt(EIGHT_PI2 / (2 * np.arange(L) + 1))
    out[N - 1 - s] = weight[:, None] * flm * lm_mask(L, s)
    return out


def project(flmn, s):
    """The adjoint projection in harmonic space (Lemma IV.7(c), Eq. 38): it reads the column n = -s."""
    N, L = n_of(flmn), flmn.shape[-2]
    if abs(s) >= N:
        return np.zeros((L, 2 * L - 1), np.complex128)
    weight = parity(s) * np.sqrt((2 * np.arange(L) + 1) / EIGHT_PI2)
    return weight[:, None] * flmn[N - 1 - s] * lm_mask(L, s)


def lift_grid(f, s, N):
    """The lift on samples (Eq. 37): (2 pi)^{-1/2} e^{-is gamma} f(beta, alpha) on the SO(3) grid."""
    return (
        np.exp(-1j * s * gamma_grid(N))[:, None, None]
        * np.asarray(f)[None]
        / np.sqrt(2 * np.pi)
    )


def project_grid(F, s):
    """The adjoint projection on samples (Eq. 37): the fibre integral against e^{is gamma}.

    The trapezoidal rule is exact for fibres band-limited at N > |s|.
    """
    N = (F.shape[0] + 1) // 2
    weights = (
        np.exp(1j * s * gamma_grid(N)) * (2 * np.pi / (2 * N - 1)) / np.sqrt(2 * np.pi)
    )
    return np.einsum("j,jtp->tp", weights, F)


# ------------------------------------------------------------------ the lifted convolution


def group_conv(block, flmn, N_out):
    """The group convolution in Wigner space (Eq. 20): (Psi * F)^l_{mn} = sum_k F^l_{mk} Psi^{l*}_{nk}."""
    N_in, L = n_of(flmn), flmn.shape[-2]
    out = np.zeros((2 * N_out - 1, L, 2 * L - 1), np.complex128)
    for n in range(-(N_out - 1), N_out):
        for k in range(-(N_in - 1), N_in):
            out[N_out - 1 + n] += (
                flmn[N_in - 1 + k] * np.conj(block[:, L - 1 + n, L - 1 + k])[:, None]
            )
    return out


def lifted_conv(block, x, s_in, s_out, N_out=None):
    """C_{s_in -> s_out}[Psi] of Definition V.1 as the composition of lift, group convolution and projection."""
    L = block.shape[0]
    F = x if s_in == "G" else lift(x, s_in, abs(s_in) + 1)
    if s_out == "G":
        return group_conv(block, F, L if N_out is None else N_out)
    return project(group_conv(block, F, abs(s_out) + 1), s_out)


def theorem_v2(block, x, s_in, s_out, N_out=None):
    """The closed forms (43)-(46) of Theorem V.2; outputs on the group keep the orders |n| < N_out (default L)."""
    L = block.shape[0]
    el = np.arange(L)
    N_out = L if N_out is None else N_out
    if s_in != "G" and s_out != "G":
        lower = max(abs(s_in), abs(s_out))
        c = (
            parity(s_in + s_out)
            * np.conj(block[:, L - 1 - s_out, L - 1 - s_in])
            * (el >= lower)
        )
        return c[:, None] * x * lm_mask(L, lower)
    if s_in != "G":
        column = np.conj(block[:, L - N_out : L + N_out - 1, L - 1 - s_in]).T
        weight = parity(s_in) * np.sqrt(EIGHT_PI2 / (2 * el + 1))
        return column[:, :, None] * (weight[:, None] * x * lm_mask(L, s_in))[None]
    N = n_of(x)
    if s_out != "G":
        row = np.conj(block[:, L - 1 - s_out, L - N : L + N - 1])
        weight = parity(s_out) * np.sqrt((2 * el + 1) / EIGHT_PI2)
        return weight[:, None] * np.einsum("klm,lk->lm", x, row) * lm_mask(L, s_out)
    return np.einsum(
        "klm,lnk->nlm",
        x,
        np.conj(block[:, L - N_out : L + N_out - 1, L - N : L + N - 1]),
    )


def lifted_conv_literal(block, samples, s_in, s_out, N):
    """C_{s_in -> s_out}[Psi] on samples: lift on the SO(3) grid, Wigner transform, block product (20),
    inverse Wigner transform and fibre integral.

    The kernel must be azimuthally band-limited on both sides, Psi^l_{nk} = 0 for |n| >= N or
    |k| >= N; inputs and outputs on the group are sampled at N.
    """
    L = block.shape[0]
    N_in = N if s_in == "G" else abs(s_in) + 1
    F = wigner_fwd(samples if s_in == "G" else lift_grid(samples, s_in, N_in), L, N_in)
    G = wigner_inv(group_conv(block, F, N), L, N)
    return G if s_out == "G" else project_grid(G, s_out)


# --------------------------------------------------------------------- kernels as blocks


def block_lifted(psi_lm, s):
    """Lambda_s psi as a kernel (Lemma IV.7(c)): Psi^l_{n,-s} = (-1)^s sqrt(8 pi^2/(2l+1)) psi_{ln}."""
    L = psi_lm.shape[-2]
    block = np.zeros((L, 2 * L - 1, 2 * L - 1), np.complex128)
    weight = parity(s) * np.sqrt(EIGHT_PI2 / (2 * np.arange(L) + 1))
    block[:, :, L - 1 - s] = weight[:, None] * psi_lm * lm_mask(L, s)
    return block


def block_flip(block):
    """The flipped kernel Psi~(g) = Psi^*(g^{-1}), whose blocks are Psi^{l dagger}."""
    return np.conj(np.transpose(block, (0, 2, 1)))


def block_left(h_lm):
    """Psi_h(g) = h^*(g^{-1} eta) (Prop. VI.4): the row n = 0, (Psi_h)^l_{0k} = 2 pi sqrt(4 pi/(2l+1)) h^*_{lk}."""
    L = h_lm.shape[-2]
    block = np.zeros((L, 2 * L - 1, 2 * L - 1), np.complex128)
    weight = 2 * np.pi * np.sqrt(FOUR_PI / (2 * np.arange(L) + 1))
    block[:, L - 1, :] = weight[:, None] * np.conj(h_lm) * lm_mask(L)
    return block


def block_multiplier(c, s, sp):
    """The canonical kernel of a multiplier (Theorem V.5(a)): (Psi_T)^l_{-s',-s} = (-1)^{s+s'} c_l^*."""
    c = np.asarray(c, np.complex128)
    L = len(c)
    block = np.zeros((L, 2 * L - 1, 2 * L - 1), np.complex128)
    keep = np.arange(L) >= max(abs(s), abs(sp))
    block[:, L - 1 - sp, L - 1 - s] = np.where(keep, parity(s + sp) * np.conj(c), 0.0)
    return block


def multiplier_of(block, s, sp):
    """The multiplier c_l = (-1)^{s+s'} Psi^{l*}_{-s',-s} read by C_{s -> s'}[Psi] (Eq. 43)."""
    L = block.shape[0]
    c = parity(s + sp) * np.conj(block[:, L - 1 - sp, L - 1 - s])
    return np.where(np.arange(L) >= max(abs(s), abs(sp)), c, 0.0)


def apply_multiplier(c, flm, s, sp):
    """(T f)_{lm} = c_l f_{lm} for l >= max(|s|, |s'|) (Theorem V.5(a))."""
    L = flm.shape[-2]
    return np.asarray(c)[:, None] * flm * lm_mask(L, max(abs(s), abs(sp)))


def conjugate_lm(flm, s):
    """Coefficients of the conjugate of a spin-s field (Eq. A9): (-1)^{s+m} f^*_{l,-m}."""
    m = m_axis(flm.shape[-2])
    return np.array([parity(s + mm) for mm in m])[None, :] * np.conj(
        np.asarray(flm)[..., ::-1]
    )


# ---------------------------------------------------------------------------- equivariance


def equivariance_error(apply, f_lm, s_out, rotation, rng, n_points=40):
    """Real-space equivariance error of a map with output on the sphere.

    T(R f)(omega) against e^{-is' vartheta} (T f)(R^{-1} omega) (Eq. 12) at random points;
    apply(f) returns a function of (theta, phi).
    """
    theta, phi = random_points(rng, n_points)
    lhs = apply(rotate_flm(f_lm, rotation))(theta, phi)
    R_inv = np.swapaxes(rot_matrix(*rotation), -1, -2)
    a, b, c = euler_zyz(R_inv @ section(theta, phi))
    return rel_err(lhs, np.exp(-1j * s_out * c) * apply(f_lm)(b, a))


def equivariance_error_group(apply, x, rotation, rng, group_input=False, n_points=12):
    """Equivariance error of a map with output on SO(3): T(pi(g) x)(rho) against (T x)(g^{-1} rho)."""
    after = apply(rotate_flmn(x, rotation) if group_input else rotate_flm(x, rotation))
    before = apply(x)
    rhos = np.array([random_rotation(rng) for _ in range(n_points)]).T
    g_inv = np.swapaxes(rot_matrix(*rotation), -1, -2)
    moved = euler_zyz(g_inv @ rot_matrix(*rhos))
    return rel_err(eval_so3(after, *rhos), eval_so3(before, *moved))
