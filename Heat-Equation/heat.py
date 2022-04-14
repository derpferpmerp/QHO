import mpmath
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from mpmath import exp
from mpmath import inf as mpINF
from mpmath import nsum as infiniteSum
from mpmath import sin
from numpy import pi as PI
from scipy.integrate import quad as integrate
from scipy.interpolate import griddata
from tqdm import tqdm

import plotly.graph_objects as go


class Heat(object):
    def __init__(self, k: np.float64 = None, L: np.float64 = None, FUNC=None):
        self.L = L
        self.K = k
        self.FUNC = FUNC

        if L is None:
            self.L = PI * 2
        if k is None:
            self.K = 0.284
        if FUNC is None:
            self.FUNC = lambda x: 6 * sin(PI*x/self.L)

    def _beta_int(self, x, n: int):
        return self.FUNC(x) * sin(n * PI * x / self.L)

    def beta(self, n: int):
        def integralFunc(x): return self._beta_int(x, n)
        integralRes = integrate(integralFunc, 0, self.L)[0]
        gamma = 2/self.L
        return gamma * integralRes

    def _func_sum(self, x, t, n):
        const = n * PI / self.L
        first = sin(const * x)
        second = exp(-self.K * t * pow(const, 2))
        return first * second

    def func_u(self, x, t):
        SUM = float(infiniteSum(
            lambda n: self._func_sum(x, t, n),
            [1, 25]
        ))
        return SUM

    def generate(self, X_LIST, TIME_LIST):
        POINTS = []
        with tqdm(total=len(X_LIST)*len(TIME_LIST)) as pbar:
            for x in X_LIST:
                for t in TIME_LIST:
                    POINTS.append([x, t, self.func_u(x, t)])
                    pbar.update(1)
        POINTS = np.array(POINTS)
        return POINTS


HeatItem = Heat()

X_L = np.linspace(0, 1, 30)
T_L = np.linspace(0, 0.5, 30)

PNTS = HeatItem.generate(X_L, T_L)

PNTS_X, PNTS_T, PNTS_Z = [
    PNTS[:, 0],
    PNTS[:, 1],
    PNTS[:, 2]
]

MESH_X, MESH_T = np.meshgrid(PNTS_X, PNTS_T)

MESH_Z = griddata(
    (PNTS_X, PNTS_T),
    PNTS_Z,
    (MESH_X, MESH_T),
    method='cubic'
)


fig = go.Figure(
    data=[
        go.Surface(
            x = MESH_X,
            y = MESH_T,
            z = MESH_Z,
        )
    ],
)

fig.update_layout(
    scene=dict(
        xaxis_title="x",
        yaxis_title="τ",
        zaxis_title="μ",
    ),
    title='The Relativistic Heat Equation',
    template="plotly_dark",
)

fig.write_image("heat_plty.png", engine="kaleido")