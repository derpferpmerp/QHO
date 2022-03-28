from math import factorial
from pprint import pprint

import numpy as np
from numpy import exp, sqrt
from scipy.constants import hbar as HBARCONST
from scipy.constants import pi as PI
from scipy.special import hermite
from matplotlib import pyplot as plt

import cmath
from celluloid import Camera
from tqdm import tqdm

plt.style.use("dark_background")

fig, ax = plt.subplots(1, 1, figsize=(10,10))
camera = Camera(fig)
SPEED_MOD = 1 # 2: Double the Video Speed
INTERPOLATE = False
GRAPH_PROB = True


class QHO(object):
    def __init__(self, mass:np.float64=None, spring_coeff:np.float64=None, hbar=None):
        
        if mass is None: mass = 0.1
        if spring_coeff is None: spring_coeff = 0.5
        if hbar is None: hbar = 1/(2*PI)
        
        self.mass = mass
        self.hbar = hbar
        self.k = spring_coeff
        self.omega = sqrt(self.k / self.mass)
        self.const = (self.mass * self.omega) / self.hbar
        
    def _calc_hermite(self, x:np.float64, n:int):
        const = sqrt(self.const) * x
        herm = hermite(n)
        return herm( const )
    
    def _calc_exp(self, x:np.float64):
        return exp( -0.5 * self.const * x * x )
    
    def _calc_quart(self):
        return pow(1/PI * self.const, 0.25)
    
    def _calc_front(self, n:int):
        return pow( pow(2,n) * factorial(n) , -0.5 )
    
    def _calc_energy(self, n:int):
        return (2*n+1) * self.hbar * 0.5 * self.omega
    
    def _compile(self, x:np.float64, n:int):
        HERM = self._calc_hermite(x, n)
        EXP = self._calc_exp(x)
        QUART = self._calc_quart()
        FRONT = self._calc_front(n)
        return HERM * EXP * QUART * FRONT
    
    def _group_compile(self, x_l:np.ndarray, n:int):
        lst = []
        for x in x_l:
            lst.append(self._compile(x, n))
        return np.array(lst)
    
    def _compute(self, x_l:np.ndarray, qnum_max=5):
        quantum = np.arange(1, qnum_max, 1)
        outlst = [ self._group_compile(x_l, 0) ]
        energies = [ self._calc_energy(0) ]
        for num in quantum:
            outlst.append( self._group_compile(x_l, num) )
            energies.append( cmath.exp( -1j * self._calc_energy(num) / self.hbar ) )
        outlst = np.array(outlst)
        energies = np.array(energies)
        return outlst, energies
    
    def _combine_solution(self, time:np.float64, x_l:np.ndarray, qnum_max=5):
        sol = self._compute(x_l, qnum_max)
        power = pow( sol[1], time )
        SOLUTIONS = []
        for pw, sl in zip(power, sol[0]):
            SOL = pow(sl, pw)
            SOLUTIONS.append(SOL)
        SUM = SOLUTIONS[0]
        for lst in SOLUTIONS[1:-1]:
            SUM  += lst
        return SUM
    
    def solve(self, t:np.ndarray, x_l:np.ndarray, qnum_max:int=5):
        SOLUTIONS = []
        for time in tqdm(t):
            SOLUTIONS.append(self._combine_solution(time, x_l, qnum_max))
        return np.array(SOLUTIONS)
    
#plt.ylim(-1,1)

def interpolate(lst:np.ndarray, MIN=0, MAX=1):
    return np.interp(
        lst,
        ( lst.min(), lst.max() ),
        ( MIN, MAX )
    )
    


if __name__ == '__main__':
    qho = QHO(mass=1, spring_coeff=4)
    X_L = np.linspace(-1, 1, 100)
    TIME = np.linspace(0, 25, 200)
    lst = qho.solve(TIME, X_L)
    for itm in tqdm(lst):
        if GRAPH_PROB:
            j = abs(itm)*abs(itm)
            J = interpolate(j)
            ax.plot(X_L, J, color="red")
        else:
            PARTS = [ [ITM.real, ITM.imag] for ITM in itm]
            REAL, IMAG = zip(*PARTS)
            COMP = []
            for re, im in PARTS:
                COMP.append(sqrt(re*re + im*im))
            if INTERPOLATE:
                REAL = interpolate(np.array(REAL))
                IMAG = interpolate(np.array(IMAG))
                COMP = interpolate(np.array(COMP))
            ax.plot(X_L, COMP, color="red")
            ax.plot(X_L, REAL, color="blue")
            ax.plot(X_L, IMAG, color="orange")
        camera.snap()
    FPS = len(TIME)/max(TIME) * SPEED_MOD
    anim = camera.animate(blit=True,) #interval=MS_PER_FRAME)
    anim.save("scatter.mp4", fps=FPS, writer="ffmpeg")
    
