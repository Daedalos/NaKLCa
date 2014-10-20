import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# The currents and parameters are taken from the NaKL model in BioCyb
# 1. As was done in that paper tau_x(V) and x_inf(V) use the same
# parameters for V_a and dV_a

# The CaL current is taken from Nogaret's notes, however I changed the
# [Ca]_in to a dynamical variable (instead of fixed) and used the
# exact GHK rather than the Taylor series because Scipy can handle the
# the 0/0 divergence at V=0.

# The I_SK is taken from Arij's model. 

gNa = 120.0 # mS/cm^2
gK = 20.0 
gL = .30 
#gSK = 0.0
#gCaL = 0.0
gCaL = 0.005 # mS/cm^2/uM(micromolar) - This is different because uses GHK form of current
gSK = 9.1 

ENa = 50.0 #mV
EK = -77.0 #mV
EL = -54.4 #mV

# The form of the sigmoid funcions is f(V) = 1/(1+exp((V-theta)/sigma))
# this is = 0.5(1+tanh((V-th)/(-2*sig)));

# In BioCyb V_a = th_a; dV_a = -2*sig_a

th_m = -40.0 #mV
th_h = -60.0 #mV
th_n = -55.0 #mV
th_s = -13.32 #mV
sig_m = -7.5 #mV
sig_h = 7.5 #mV
sig_n = -15.0 #mV
sig_s = -8.0 #mV

t0_m = 0.1 #ms
t0_h = 1.0 #ms
t0_n = 1.0 #ms
t0_s = 0.2 #ms

t1_m = 0.4 #ms
t1_h = 7.0 
t1_n = 5.0
t1_s = 9.0 # s-variable was not originally included in NaKL (came from Arij). Constant tau_s = t0_s seems as valid as anything else


CaExt = 2500.0 # uM
k_s = 0.5 # uM

eps = 7.5e-3 # uM*cm^2/uA/ms
kCa = 0.3 # uM
bCa = 0.24 # uM
C = 1.0 # uF/cm^2


#PROBABLY NOT for estimation
Isa = 1.0e6 #uA/pA. Scaling for Iinj: Iinj/Isa should be in [uA]
f = 0.1 #unitless. Fraction of Free/Total cytosolic Ca^2+. Degenerate with params kCa, bCa.
p = 2.0 #unitless. Exponent of hill fcn. Hardcoded in Arij model
T = 290.0 #K . Temperature, Probably doesn't need to be estimated

def I_Na(V,m,h):        
    return gNa * m**3 * h * (ENa-V)

def I_K(V,n):
    return gK * n**4 * (EK-V)

#note, this is leak current, not L-type Ca
def I_L(V):
    return gL * (EL-V)

# This is SK current from Arij's model. I *think* SK is a specific
# class of Ca-gated K currents
def I_SK(V, Ca):
    kinf = Ca**p/(Ca**p + k_s**p)
    return gSK * kinf * (EK-V)

def I_CaL(V, Ca,s):
    #23.209 is 2*F/R in K/mV
    return gCaL * V * s**2 *(Ca-CaExt*sp.exp(-23.209*V/T))/(sp.exp(-23.209*V/T)-1)
#    return gCaL * V * s**2 * (CaExt)/(1-sp.exp(23.209*V/T))

def gating_inf(V,theta,sigma): 
    # Using exponential form for straightforward comparison with other
    # neuron models/parameters
    return 1.0/(1+sp.exp((V-theta)/sigma))

def tau(V,t0,t1,theta,sigma):
    # I am using the exponetial form of x_inf(V), so there's a
    # relative factor of (-2) in the sigma.
    return t0 + t1*(1.0-sp.tanh((V-theta)/(-2.0*sigma))**2)    
    
# Give this function an array of current values and the time step
# between them, it will interpolate using a simple line between
# neighboring time points
# Current is an array of current data points, seperated by "step" amount of time
def Iinj(t, Ivalue, step): 
    idx = int(t/step)
    if idx == t/step:
        return Ivalue[idx]
    else:
        slope = (Ivalue[idx+1]-Ivalue[idx])/step               
        return Ivalue[idx] + slope*(t-idx*step)    

def eqns(x, t, current, tstep):
    V, Ca, m, h, n, s = x            

    dVdt = 1.0/C * (I_Na(V,m,h) + I_K(V,n) + I_L(V) + I_SK(V,Ca) + I_CaL(V,Ca,s) + Iinj(t,current,tstep))

    dCadt = f*(eps*I_CaL(V,Ca,s) + kCa*(bCa - Ca))     
    dmdt = (gating_inf(V,th_m,sig_m)-m)/ tau(V,t0_m,t1_m,th_m,sig_m)
    dhdt = (gating_inf(V,th_h,sig_h)-h)/ tau(V,t0_h,t1_h,th_h,sig_h)
    dndt = (gating_inf(V,th_n,sig_n)-n)/ tau(V,t0_n,t1_n,th_n,sig_n)
    dsdt = (gating_inf(V,th_s,sig_s)-s)/ tau(V,t0_s,t1_s,th_s,sig_s)

    return dVdt, dCadt, dmdt, dhdt, dndt, dsdt
    

def run():
    
    # Initial Conditions
    init = [-70.0, 1.0, 0.1, 0.1, 0.1, 0.1]
    # Total Integration time and 
    T = 100.0
    # output time steps
    dt = 0.01

    #Injected current file. Also need to specify time-step between
    #data points in file. Will linearly interpolate for intermediate
    #points
    inj = sp.loadtxt('current_sq.txt')
    injdt = 2.0    
    #inj = sp.loadtxt('current_l63.txt')
    #injdt = 0.01

    times = sp.arange(0,T,dt)

    # Parameters passed are current data array, along with time step
    # between current data points
    sim = odeint(eqns,init,times,(inj,injdt))
    sp.savetxt('simulation.txt',sp.column_stack((times,sim)))


    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(times,sim[:,0])
    ax[0,0].set_title('Voltage')
    ax[0,1].plot(times,sim[:,1])
    ax[0,1].set_title('Calcium')
    ax[1,0].plot(times,sim[:,2])
    ax[1,0].set_title('m (Na act)')
    ax[1,1].plot(times,sim[:,3])
    ax[1,1].set_title('h (Na inact)')
    ax[2,0].plot(times,sim[:,4])
    ax[2,0].set_title('n (K act)')
    ax[2,1].plot(times,sim[:,5])
    ax[2,1].set_title('s (CaL act)')

    fig2, ax2 = plt.subplots(3,2)
    ax2[0,0].plot(times,I_Na(sim[:,0],sim[:,2],sim[:,3]))
    ax2[0,0].set_title('I_Na (\mu A/cm^2)')
    ax2[0,1].plot(times,I_K(sim[:,0],sim[:,4]))
    ax2[0,1].set_title('I_K (\mu A/cm^2)')
    ax2[1,0].plot(times,I_L(sim[:,0]))
    ax2[1,0].set_title('I_L (\mu A/cm^2)')
    ax2[1,1].plot(times,I_SK(sim[:,0], sim[:,1]))
    ax2[1,1].set_title("I_SK (\mu A/cm^2)")
    ax2[2,0].plot(times,I_CaL(sim[:,0],sim[:,1],sim[:,5]))
    ax2[2,0].set_title("I_CaL (\mu A/cm^2)")             

    ttmp = sp.arange(0,T,injdt)
    ax2[2,1].plot(ttmp, inj[:len(ttmp)]/Isa)
    ax2[2,1].set_title("I_Inj")

    fig.subplots_adjust(hspace=.5)
    fig2.subplots_adjust(hspace=.5)
    plt.show()
        
if __name__ == '__main__':
    run()
    
