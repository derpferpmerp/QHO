from numpy import linspace, zeros, log, float64, meshgrid
from utils import convertPat, hidePlotBounds, pathfix
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 1, figsize=(10,10))

class Lyapunov(object):
    def __init__(self, pat:str=None, iterations:int=None):
        if pat is None: self.pat = convertPat("AABAB")
        else: self.pat = convertPat(pat)
        if iterations is None: self.itrmx = 100
        else: self.itrmx = iterations
        
    def getVal(self, indx):
        return self.pat[ indx % len(self.pat) ]
        
    def _pick(self, s, a, b):
        if s: return a
        else: return b
        
    def lyap_inst(self, X, indx, a, b):
        CHR = self.getVal(indx)
        r_n = self._pick(CHR, a, b)
        return [
            r_n,
            r_n * X * (1 - X)
        ]
        
    def _compute_lst(self, a, b):
        X = [ [0, 0.5] ]
        for itr in range(self.itrmx):
            XN = X[-1][1]
            X.append( self.lyap_inst( XN, itr, a, b ) )
        return X

    def _compute_pSum(self, r_n, x_n):
        term = abs(r_n * (1 - 2*x_n))
        return term if term == 0 else log( term )

    def check(self, itm):
        return itm is None

    def compute_sum(self, a, b):
        ADD = [
            self._compute_pSum( r_n, x_n )
            for r_n, x_n in self._compute_lst(a, b)[1:]
        ]
        return 0.01 if None in ADD else sum(ADD)/self.itrmx

    def create_grid(self, x_l, y_l):
        GRID = zeros(
            ( len(x_l), len(y_l) ),
            dtype=float64
        )
        with tqdm(total=len(x_l)*len(y_l)) as pbar:
            for i_x, x in enumerate(x_l):
                for i_y, y in enumerate(y_l):
                    GRID[ i_x, i_y ] = self.compute_sum(x, y)
                    pbar.update(1)
        return GRID

X_VIEW = (3.4, 4)
Y_VIEW = (2.5, 3.4)
RAT_MULT = 500
ITERATIONS = 300
PAT = "BBBBBBAAAAAA"

rat_x = round(RAT_MULT*abs(X_VIEW[1] - X_VIEW[0]))
rat_y = round(RAT_MULT*abs(Y_VIEW[1] - Y_VIEW[0]))

print(
    "-- INPUT --",
    f"X Pixels: {rat_y}",
    f"Y Pixels: {rat_x}",
    f"ViewBox: {X_VIEW}, {Y_VIEW}",
    "-- --",
    sep="\n"
)

GX = linspace(*X_VIEW, rat_x)
GY = linspace(*Y_VIEW, rat_y)


CLS = Lyapunov(iterations=ITERATIONS, pat=PAT)
GRID = CLS.create_grid(GX, GY)

im = ax.imshow(
    GRID,
    cmap="hsv",
    interpolation="gaussian"
)

hidePlotBounds(ax)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)

plt.savefig(
    pathfix("out.png"),
    bbox_inches="tight",
    pad_inches=1,
    format="png",
)		