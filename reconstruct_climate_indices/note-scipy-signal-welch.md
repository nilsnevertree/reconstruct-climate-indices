## How [scipy.signal.welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html) calculates PSD

Here is a small explanation how the scipy.signal.welch function computes PSD

$x = x(t)$ with length $N=T/dt$

If we use a simple boxcar window, the single components are $win = 1/N$.
This calls the normal DFT function as in wikipedia

$ x(f) = \sum_{j=0}^{N} x_j \cdot exp\left(-2 i \pi f j \right)$

On [scipy.fft.fft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#scipy.fft.fft) this is the same formular but with $k = fN$.

Now the scipy.signal.welch function calls [_spectral_py](https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/signal/_spectral_py.py#L1631).

But hands it:

$ win \cdot x(t) = 1/N \cdot x(t)$

Thus the result is not purely the DFT but:

$result = 1/N \cdot \hat{x}(f)$

The scaling is ``1/(fs (win*win).sum())``. ``(win*win).sum() = 1/N`` for the boxcar window.

$PSD = dt \cdot N \cdot \left|\frac{1}{N} \hat{x}(f)\right|^2 $

Now we knoe that $N = T/dt$.

$PSD = \frac{dt}{N} \cdot \left|\hat{x}(f)\right|^2 $

So we get finally:
$$PSD = \frac{dt^2}{T} \cdot \left|\hat{x}(f)\right|^2 $$
