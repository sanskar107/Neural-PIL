from typing import Optional
import tensorflow as tf
import numpy as np

EPS = 1e-7


def magnitude(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("magnitude"):
        return safe_sqrt(dot(x, x))



def safe_sqrt(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("safe_sqrt"):
        sqrt_in = tf.maximum(x, EPS)
        return tf.sqrt(sqrt_in)


def safe_log(x):
    """The same as tf.math.log(x), but clamps the input to prevent NaNs."""
    return tf.math.log(tf.minimum(x, fill_like(x, 33e37)))


def safe_log1p(x):
    """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
    return tf.math.log1p(tf.minimum(x, fill_like(x, 33e37)))


def safe_exp(x):
    """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
    return tf.math.exp(tf.minimum(x, fill_like(x, 87.5)))


def safe_expm1(x):
    """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
    return tf.math.expm1(tf.minimum(x, fill_like(x, 87.5)))


def dot(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("dot"):
        return tf.reduce_sum(x * y, axis=-1, keepdims=True)

def spherical_to_cartesian(theta: tf.Tensor, phi: tf.Tensor, r=1) -> tf.Tensor:
    """
    The referrence plane is the Cartesian xy plane
    phi is inclination from the z direction.
    theta is measured from the Cartesian x axis (so that the axis has theta = +90°).

    Args:
        theta: is azimuth [0, 2pi)
        phi: is inclination [0, Pi],
        r: is length  [0, inf)

    Returns:
        The cartesian vector (x,y,z)
    """
    y = r * tf.sin(phi) * tf.sin(theta)
    z = r * tf.cos(phi)
    x = -r * tf.sin(phi) * tf.cos(theta)

    return tf.concat([x, y, z], -1)


def cartesian_to_spherical(vec: tf.Tensor) -> tf.Tensor:
    """
    Args:
        vec: cartesian vector

    Returns:
        Theta is azimuth [0, 2pi)
        Phi is inclination [0, Pi],
        r is length [0, inf)
    """
    x, y, z = vec[..., 0:1], vec[..., 1:2], vec[..., 2:3]

    r = magnitude(vec)
    theta = tf.math.atan2(y, -x)
    # atan2 outputs value from -pi to pi.
    # We expect 0 to 2pi.
    # The negative values need to map to second quadrant
    theta = tf.where(theta > 0, theta, 2 * np.pi + theta)
    # Lastly theta should never reach 2pi so wrap around
    theta = tf.math.floormod(theta, 2 * np.pi - EPS)

    # Phi is just acos and safety to avoid div by 0
    phi = tf.math.acos(tf.clip_by_value(tf.math.divide_no_nan(z, r), -1, 1))

    return theta, phi, r


def spherical_to_uv(spherical: tf.Tensor) -> tf.Tensor:
    # Turst no one
    theta = tf.clip_by_value(spherical[..., 0], 0, 2 * np.pi - EPS,)  # [0, 2pi)
    phi = tf.clip_by_value(spherical[..., 1], 0, np.pi)  # [0, pi]

    u = theta / (2 * np.pi)  # TODO this might be a np.fliplr candidate
    v = phi / np.pi

    return tf.math.abs(tf.stack([u, v], -1))


def direction_to_uv(d: tf.Tensor) -> tf.Tensor:
    theta, phi, r = cartesian_to_spherical(d)
    return spherical_to_uv(tf.concat([theta, phi], -1))


def uv_to_spherical(uvs: tf.Tensor) -> tf.Tensor:
    # Turst no one
    u = tf.clip_by_value(uvs[..., 0], 0, 1)  # TODO this might be a np.fliplr candidate
    v = tf.clip_by_value(uvs[..., 1], 0, 1)

    theta = tf.clip_by_value((2 * u * np.pi), 0, 2 * np.pi - EPS,)  # [-pi, pi)
    phi = tf.clip_by_value(np.pi * v, 0, np.pi)  # [0, pi]

    return tf.stack([theta, phi], -1)


def uv_to_direction(uvs: tf.Tensor) -> tf.Tensor:
    spherical = uv_to_spherical(uvs)
    theta = spherical[..., 0:1]
    phi = spherical[..., 1:2]
    return spherical_to_cartesian(theta, phi)


def shape_to_uv(height: int, width: int) -> tf.Tensor:
    # UV
    # 0,0              1,0
    # 0,1              1,1
    us, vs = tf.meshgrid(
        tf.linspace(
            0.0 + 0.5 / tf.cast(width, tf.float32),
            1.0 - 0.5 / tf.cast(width, tf.float32),
            width,
        ),
        tf.linspace(
            0.0 + 0.5 / tf.cast(height, tf.float32),
            1.0 - 0.5 / tf.cast(height, tf.float32),
            height,
        ),
    )  # Use pixel centers
    return tf.cast(tf.stack([us, vs], -1), tf.float32)


# dirs = uv_to_direction(shape_to_uv(4, 4))
# print(dirs)
# print(shape_to_uv(4, 4))

dirs = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [0.25, -0.25, 0.25]
    ]
)
dirs = tf.convert_to_tensor(dirs, dtype=tf.float32)
uv = direction_to_uv(dirs)
print(uv)

print(uv_to_direction(uv))