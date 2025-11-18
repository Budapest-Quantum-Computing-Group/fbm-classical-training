import jax
import jax.numpy as jnp

from jax import lax

import numpy as np

from functools import partial

import itertools

from src.SO import calculate_SO

from src.pfaffian import pfaffian

from src.indices import four_ary_array

from scipy.special import comb

_ = 0


@jax.jit
def get_base_covariance_matrix_on_subsystem(alpha):
    r"""
    The covariance matrix responsinble for the 0, 2, and 6-length majorana strings.

    Args:
        alpha (float): The single parameter corresponding to the subsystem.
    """
    cos_2_alpha = jnp.cos(2 * alpha)

    J = jnp.array([[0, cos_2_alpha], [-cos_2_alpha, 0]], dtype=complex)

    return jax.scipy.linalg.block_diag(*([J] * 4))


@jax.jit
def get_00_covariance_matrix_on_subsystem(alpha):
    normalization = jnp.cos(alpha) ** 2 + 1

    a = -jnp.sin(alpha) ** 2 / normalization
    b = -2 * jnp.cos(alpha) / normalization

    matrix = jnp.array(
        [
            [_, a, _, -b, _, _, _, _],
            [-a, _, -b, _, _, _, _, _],
            [_, b, _, a, _, _, _, _],
            [b, _, -a, _, _, _, _, _],
            [_, _, _, _, _, 1, _, _],
            [_, _, _, _, -1, _, _, _],
            [_, _, _, _, _, _, _, 1],
            [_, _, _, _, _, _, -1, _],
        ],
        dtype=complex,
    )

    return matrix


@jax.jit
def get_11_covariance_matrix_on_subsystem(alpha):
    normalization = jnp.sin(alpha) ** 2 + 1

    a = jnp.cos(alpha) ** 2 / normalization
    b = -2 * jnp.sin(alpha) / normalization

    matrix = jnp.array(
        [
            [_, -1, _, _, _, _, _, _],
            [1, _, _, _, _, _, _, _],
            [_, _, _, -1, _, _, _, _],
            [_, _, 1, _, _, _, _, _],
            [_, _, _, _, _, a, _, b],
            [_, _, _, _, -a, _, b, _],
            [_, _, _, _, _, -b, _, a],
            [_, _, _, _, -b, _, -a, _],
        ],
        dtype=complex,
    )

    return matrix


@jax.jit
def get_01_covariance_matrix_on_subsystem(alpha):
    a = 1
    b = -1

    c = jnp.cos(alpha)
    s = jnp.sin(alpha)

    jc = 1j * c
    js = 1j * s

    matrix = jnp.array(
        [
            [_, -a, -jc, c, _, _, _, _],
            [a, _, c, jc, _, _, _, _],
            [jc, -c, _, -a, _, _, _, _],
            [-c, -jc, a, _, _, _, _, _],
            [_, _, _, _, _, -b, js, -s],
            [_, _, _, _, b, _, -s, -js],
            [_, _, _, _, -js, s, _, -b],
            [_, _, _, _, s, js, b, _],
        ],
        dtype=complex,
    )

    return matrix


@jax.jit
def get_10_covariance_matrix_on_subsystem(alpha):
    return jnp.conj(get_01_covariance_matrix_on_subsystem(alpha))


@jax.jit
def _embed_and_evolve(covariance_matrix, subsystem_index, SO) -> jnp.ndarray:
    """
    Compute  Σ_out = SO · diag(M, 0, …, 0) · SOᵀ
    without ever materialising the big block-diagonal matrix.

    Returns a (2N, 2N) covariance matrix living in the same subsystemtor-space as `SO`.
    """
    blk = 8
    n_tot = SO.shape[0]
    start = subsystem_index * blk

    B = lax.dynamic_slice(SO, (0, start), (n_tot, blk))

    A = lax.dynamic_slice(B, (start, 0), (blk, blk))

    M = covariance_matrix
    M_AT = M @ A.T
    C_M_AT = B @ M_AT

    Σ11 = A @ M_AT
    Σ12 = -C_M_AT.T
    Σ22 = B @ M @ B.T

    Σ = Σ22
    Σ = lax.dynamic_update_slice(Σ, Σ11, (start, start))
    Σ = lax.dynamic_update_slice(Σ, Σ12, (start, 0))
    Σ = lax.dynamic_update_slice(Σ, C_M_AT, (0, start))
    return Σ


@jax.jit
def _get_single_Z_string_expval(Z_string, covariance_matrix):
    submatrix = covariance_matrix[jnp.ix_(Z_string, Z_string)]
    pf = jnp.real(pfaffian(submatrix))
    return pf


@jax.jit
def _get_Z_string_expvals_from_covariance_matrix(covariance_matrix, Z_strings):
    return jax.vmap(_get_single_Z_string_expval, in_axes=(0, None))(
        Z_strings, covariance_matrix
    )


@jax.jit
def _normalisations_from_alphas(alphas: jnp.ndarray) -> jnp.ndarray:
    cos2 = jnp.cos(alphas) ** 2 + 1
    sin2 = jnp.sin(alphas) ** 2 + 1
    minus1 = -jnp.ones_like(alphas)
    return jnp.stack([cos2, sin2, minus1, minus1])


_COV_FNS = (
    get_00_covariance_matrix_on_subsystem,  # type 0
    get_11_covariance_matrix_on_subsystem,  # type 1
    get_01_covariance_matrix_on_subsystem,  # type 2
    get_10_covariance_matrix_on_subsystem,  # type 3
)


@jax.jit
def _build_components(alphas: jnp.ndarray, SO: jnp.ndarray):
    type_axis = jnp.arange(4)
    subs_axis = jnp.arange(alphas.shape[0])

    def _one(type_idx, alpha, i):
        M = jax.lax.switch(type_idx, _COV_FNS, alpha)

        return _embed_and_evolve(M, i, SO)

    return jax.vmap(jax.vmap(_one, in_axes=(None, 0, 0)), in_axes=(0, None, None))(
        type_axis, alphas, subs_axis
    )


@jax.jit
def _get_base_covariance_matrices(alphas, SO):
    N = SO.shape[0] // 8
    subsystem_indices = jnp.arange(N)

    def _get_single_base_covariance_matrix(alpha, i):
        M_i = get_base_covariance_matrix_on_subsystem(alpha)
        return _embed_and_evolve(M_i, i, SO)

    return jax.vmap(_get_single_base_covariance_matrix, in_axes=(0, 0))(
        alphas, subsystem_indices
    )


def _get_combinatorial_factors(N, max_Z_locality):
    combinatorial_factors = jnp.zeros(
        shape=(max_Z_locality // 2 + 1, max_Z_locality + 1)
    )

    for q in range(max_Z_locality // 2 + 1):
        for i in range(2 * q, max_Z_locality + 1):
            combinatorial_factors = combinatorial_factors.at[q, i].set(
                ((-1) ** ((i - 2 * q) // 2) * comb(N - 1 - q, (i - 2 * q) // 2))
                if N - q > 0
                else 1.0
            )

    return combinatorial_factors


@partial(jax.jit, static_argnums=(0, 1))
def _get_combinations(n: int, l: int) -> jnp.ndarray:
    combos_np = np.array(list(itertools.combinations(range(n), l)), dtype=np.int32)
    return jnp.asarray(combos_np, dtype=np.int32)


@jax.jit
def _construct_comp_contrib(comps, key, subsystem):
    T, N, d, _ = comps.shape

    lin = key * N + subsystem  # (q,)

    comps_flat = comps.reshape(T * N, d, d)  # (T*N, d, d)
    picked = jnp.take(comps_flat, lin, axis=0)  # (q, d, d)

    return picked.sum(axis=0)  # (d, d)


def calculate_Z_string_expvals(weights, Z_strings, N):
    """
    Calculate Z string expectation values for general FBMs.

    Args:
        weights: Model parameters (alphas and thetas).
        Z_strings: List of Z strings to calculate expectation values for.
        N: Number of subsystems.

    Returns:
        Z string expectation values.
    """

    max_Z = len(Z_strings) - 1

    Z_strings = [jax.device_put(x) for x in Z_strings]

    ssy_tables = {}
    key_tables = {}
    for no_subsystems in range(1, max_Z // 2 + 1):
        ssy_tables[no_subsystems] = _get_combinations(N, no_subsystems)
        key_tables[no_subsystems] = four_ary_array(no_subsystems)

    comb_f = _get_combinatorial_factors(N, max_Z)
    lens = jnp.array([z.shape[0] for z in Z_strings], jnp.int32)
    offs = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(lens)])
    d = 8 * N

    alphas, thetas = weights[:N], weights[N:]
    no_weights_per_layer = d * (d - 1) // 2
    no_layers = len(thetas) // no_weights_per_layer
    SO = np.identity(d)
    for i in range(no_layers):
        SO = SO @ calculate_SO(
            thetas[i * no_weights_per_layer : (i + 1) * no_weights_per_layer], d
        )

    base_cov = _get_base_covariance_matrices(alphas, SO)
    comps = _build_components(alphas, SO)
    norms = _normalisations_from_alphas(alphas).T.reshape(-1)

    cov = base_cov.sum(axis=0)

    Z_exp = jnp.concatenate(
        [
            comb_f[0, i] * _get_Z_string_expvals_from_covariance_matrix(cov, z_block)
            for i, z_block in enumerate(Z_strings)
        ]
    )

    for no_subsystems in range(1, max_Z // 2 + 1):
        subsystems = ssy_tables[no_subsystems]
        keys = key_tables[no_subsystems]

        relevant_comb_factors = comb_f[no_subsystems]

        def subsystem_body(subsystem):
            base_contrib = jnp.take(base_cov, subsystem, axis=0).sum(axis=0)
            cov_minus = cov - base_contrib

            partial_norm_index = subsystem * 4

            def key_body(key):
                comp_contrib = _construct_comp_contrib(comps, key, subsystem)
                current_cov = cov_minus + comp_contrib

                multiplier_from_normalization = jnp.prod(
                    jnp.take(norms, partial_norm_index + key, axis=0)
                )

                return jnp.concatenate(
                    [
                        (
                            jnp.take(relevant_comb_factors, i)
                            * multiplier_from_normalization
                        )
                        * _get_Z_string_expvals_from_covariance_matrix(
                            current_cov, Z_strings[i]
                        )
                        for i in range(2 * no_subsystems, max_Z + 1)
                    ]
                )

            return jnp.sum(jax.vmap(key_body)(keys), axis=0)

        Z_exp = Z_exp.at[offs[2 * no_subsystems] : offs[-1]].add(
            jnp.sum(jax.vmap(subsystem_body)(subsystems), axis=0)
        )

    return Z_exp


def calculate_Z_string_expvals_free(weights, Z_strings, N):
    """
    Calculate Z string expectation values for free-FBMs.

    Args:
        weights: Model parameters (thetas).
        Z_strings: List of Z strings to calculate expectation values for.
        N: Number of subsystems.

    Returns:
        Z string expectation values.
    """

    max_Z = len(Z_strings) - 1

    Z_strings = [jax.device_put(x) for x in Z_strings]

    comb_f = _get_combinatorial_factors(N, max_Z)
    d = 8 * N

    no_weights_per_layer = d * (d - 1) // 2
    no_layers = len(weights) // no_weights_per_layer
    SO = np.identity(d)
    for i in range(no_layers):
        SO = SO @ calculate_SO(
            weights[i * no_weights_per_layer : (i + 1) * no_weights_per_layer], d
        )

    base_cov = _get_base_covariance_matrices(np.zeros(N), SO)

    cov = base_cov.sum(axis=0)

    Z_exp = jnp.concatenate(
        [
            comb_f[0, i] * _get_Z_string_expvals_from_covariance_matrix(cov, z_block)
            for i, z_block in enumerate(Z_strings)
        ]
    )

    return Z_exp
