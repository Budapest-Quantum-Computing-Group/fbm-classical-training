"""
Copied from https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian/blob/main/py_pfaffian/jax/utils.py
"""

from jax import Array
import jax.numpy as jnp
from jax import lax
from jax import jit


from jax import custom_jvp


def pivot(_A: Array, _k: int, _kp: int):
    """Perform a Pivot on the Matrix A

    Args:
        _A (Array): Matrix, A, of size nxn where n > _k, n > _kp
        _k (int): The first index to swap
        _kp (int): The second index to swap

    Returns:
        tuple(Array, float): The Matrix with the two specified rows swapped, and the sign of the operation.
    """

    temp = _A[_k + 1]
    _A = _A.at[_k + 1].set(_A[_kp])
    _A = _A.at[_kp].set(temp)

    # Then interchange columns _k+1 and _kp
    temp = _A[:, _k + 1]
    _A = _A.at[:, _k + 1].set(_A[:, _kp])
    _A = _A.at[:, _kp].set(temp)

    return _A, -1.0


def no_pivot(_A: Array, _k: int, _kp: int):
    """A Null operation.  Mimic the signature and return of the `pivot` function.  Always returns _A, 1.0

    Args:
        _A (Array): Matrix, A, of size nxn where n > _k, n > _kp
        _k (int): Ignored
        _kp (int): Ignored

    Returns:
        tuple(Array, float): The Matrix _A, and 1.0
    """

    return _A, 1.0


def form_gauss_vector(_A: Array, _k: int):
    """
    Form the gauss vector and update the pfaffian value as the return


    Args:
        _A (Array): Matrix A, from which to create the Gauss Vector
        _k (int): The index _k of the matrix, where _k < A.shape[0]

    Returns:
        tuple(Array, Array): The updated matrix _A and the update to the pfaffian value
    """

    pfaffian_update = _A[_k, _k + 1]

    mu = _A[_k, :] / pfaffian_update
    nu = _A[:, _k + 1]

    _A = _A + jnp.outer(mu, nu) - jnp.outer(nu, mu)

    return _A, pfaffian_update

@custom_jvp
@jit
def pfaffian(A):
    """pfaffian_LTL(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.
    """

    # JAX does not support runtime checking.
    # All checks are removed!
    n = A.shape[0]
    if n == 0:
        return 1.0

    if n % 2 == 1:
        return 0.0

    pfaffian_val = 1.0

    def pfaffian_iteration(carry, _k):
        _A, _current_pfaffian_val = carry

        full_region = _A[:, _k]
        mask = jnp.arange(full_region.shape[0]) < _k
        full_region = jnp.where(mask, 0.0, full_region)

        _kp = jnp.abs(full_region).argmax()

        # Apply a pivot if needed
        _A, sign = lax.cond(_kp != _k + 1, pivot, no_pivot, _A, _k, _kp)

        _A, update = form_gauss_vector(_A, _k)

        return (_A, sign * update * _current_pfaffian_val), None

    k_list = jnp.arange(0, n - 1, 2)

    carry = (A, pfaffian_val)

    carry, _ = lax.scan(pfaffian_iteration, carry, k_list)

    return carry[1]


@pfaffian.defjvp
def pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents
    primal_out = pfaffian(A)

    # dpf(A)/dt = 1/2 pf(A) * tr(A^-1 dA/dt)
    A_inv = jnp.linalg.inv(A)

    product = jnp.matmul(A_inv, A_dot)
    product = primal_out * product

    tangent_out = 0.5 * jnp.linalg.trace(product)
    return primal_out, tangent_out

