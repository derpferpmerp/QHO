from mpmath import nsum as infiniteSum, inf as mpINF, exp, sin
import mpmath
import numpy as np
from numpy import pi as PI
from scipy.integrate import quad as integrate
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from celluloid import Camera

SPEED_MOD = 30.0
ANIMATE = False


plt.style.use("dark_background")
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax = plt.subplot(111, projection="3d")
camera = Camera(fig)

class Heat(object):
    def __init__(self, k:np.float64=None, L:np.float64=None, FUNC=None):
        self.L = L
        self.K = k
        self.FUNC = FUNC
        
        if L is None: self.L = PI * 2
        if k is None: self.K = 0.284
        if FUNC is None:
            self.FUNC = lambda x: 6 * sin(PI*x/self.L)

    def _beta_int(self, x, n:int):
        return self.FUNC(x) * sin(n * PI * x / self.L)

    def beta(self, n:int):
        integralFunc = lambda x: self._beta_int(x, n)
        integralRes = integrate(integralFunc, 0, self.L)[0]
        gamma = 2/self.L
        return gamma * integralRes

    def _func_sum(self, x, t, n):
        const = n * PI / self.L
        first = sin( const * x )
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

X_L = np.linspace(0, 1, 75)
T_L = np.linspace(0, 1, 75)

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
    method='linear'
)


Gx, Gy = np.gradient(MESH_Z) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude

INTERP = np.interp(G, (G.min(), G.max()), (0, 1))

ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$\\mu$")
plt.title("The Relativistic Heat Equation")
surf = ax.plot_surface(
    MESH_X, MESH_T, MESH_Z,
    rcount=500, ccount=500,
    facecolors=cm.jet(INTERP),
    shade=False,
    linewidth=0,
    antialiased=False,
)


if ANIMATE:
    for i in tqdm(range(360)):
        ax.view_init(elev=10., azim=i)
        camera.snap()
    FPS = SPEED_MOD
    anim = camera.animate(blit=True,) #interval=MS_PER_FRAME)
    print("Animated Frames, Saving:")
    with tqdm(total=360) as PBAR:
        anim.save(
            "HEAT.mp4",
            fps=FPS,
            writer="ffmpeg",
            progress_callback=lambda cf, tf: PBAR.update(1)
        )
else:
    plt.savefig("heat_mpl.png", bbox_inches="tight")

