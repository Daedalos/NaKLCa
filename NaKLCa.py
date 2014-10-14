import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Note: the KCa(SK) and CaL currents are taken from Arij's thesis with
#adjustments: 

#1) I made the gating variable s dynamic, rather than using the
#steady-state funciton of voltage (s_inf) as Arij does

#2) slightly modified I_CaL to include the internal Ca concentration,
#in line with canonical GHK form:
#http://en.wikipedia.org/wiki/GHK_flux_equation

#3) The signs of the currents are switched in the code, relative to
#Arij's paper, because I don't like having negative signs in front of
#all the currents in dVdt. 

#4) gCaL is 1/100th of what it is in Arij's model. I'm not sure where the discrepancy is, but gCaL = 19.0 preoduced unphysical behavior.

#5) I had to change the Conductance to be 1/10th of Arij's paper, and the conductances to be 10x Arij's paper; in order to suitably scale the inj current

### Define global parameters (probably not best practice, but I don't
### feel like passing all the arguments)

gNa = 30.0 #nS
gK = 40.0 #nS
gL = .20 #nS
#gSK = 0.0
#gCaL = 0.0
gSK = 2.7 #nS
gCaL = 0.0190 #nS . I'm not actually 100% certain of units, CaL uses GHK equation, which and gCaL just lumps all the constants out in front

ENa = 50.0 #mV
EK = -90.0 #mV
EL = -70.0 #mV
th_m = -35.0 #mV
th_h = -45.0 #mV
th_n = -30.0 #mV
th_s = -20.0 #mV
sig_m = -5.0 #mV
sig_h = 4.0 #mV
sig_n = -5.0 #mV
sig_s = -0.05 #mV
tau_m = 0.1 #ms
tau_h = 1.0 #ms
tau_n = 1.0 #ms
tau_s = 0.1 #ms
CaExt = 2500.0 # uM
k_s = 0.5 # uM
f = 0.1 #unitless
eps = 0.0015 # uM/ pA /ms
kCa = 0.3 # uM
bCa = 0.5 # uM
C = 2.0 # pF

p = 2.0 #unitless. Exponent of hill fcn. Hardcoded in Arij model
T = 290.0 #K . Probably doesn't need to be estimated

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

def m_inf(V): 
    return 1.0/(1+sp.exp((V-th_m)/sig_m))

def h_inf(V):
    return 1.0/(1+sp.exp((V-th_h)/sig_h))

def n_inf(V):
    return 1.0/(1+sp.exp((V-th_n)/sig_n))

def s_inf(V):
    #rearrange to prevent overflow. sig_s is very small
    return 1.0/(1.0+sp.exp((V-th_s)/sig_s))            
    
# Give this function an array of current values and the time step
# between them, it will interpolate using a simple line between
# neighboring time points
def Iinj(t, Ivalue, step): 
    idx = int(t/step)
    if idx == t/step:
        return Ivalue[idx]
    else:
        slope = (Ivalue[idx+1]-Ivalue[idx])/step               
        return Ivalue[idx] + slope*(t-idx*step)    

def eqns(x, t, current, tstep):
    V, Ca, m, h, n, s = x            

    dVdt = 1/C * (I_Na(V,m,h) + I_K(V,n) + I_L(V) + I_SK(V,Ca) + I_CaL(V,Ca,s) + Iinj(t,current,tstep))

    dCadt = f*(eps*I_CaL(V,Ca,s) + kCa*(bCa - Ca))     
    dmdt = (m_inf(V)-m)/ tau_m
    dhdt = (h_inf(V)-h)/ tau_h
    dndt = (n_inf(V)-n)/ tau_n
    dsdt = (s_inf(V)-s)/ tau_s

    return dVdt, dCadt, dmdt, dhdt, dndt, dsdt
    

def run():
    
    init = [-70, 1, 0.1, 0.1, 0.1, 0.1]
    T = 300; dt = 0.01

    #inj = sp.loadtxt('current_sq.txt')
    inj = sp.loadtxt('current_l63.txt')

    times = sp.arange(0,T,dt)

    sim = odeint(eqns,init,times,(inj,0.01))
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
    ax2[0,0].set_title('I_Na')
    ax2[0,1].plot(times,I_K(sim[:,0],sim[:,4]))
    ax2[0,1].set_title('I_K')
    ax2[1,0].plot(times,I_L(sim[:,0]))
    ax2[1,0].set_title('I_L')
    ax2[1,1].plot(times,I_SK(sim[:,0], sim[:,1]))
    ax2[1,1].set_title("I_SK")
    ax2[2,0].plot(times,I_CaL(sim[:,0],sim[:,1],sim[:,5]))
    ax2[2,0].set_title("I_CaL")             

    ttmp = sp.arange(0,T,dt)
    ax2[2,1].plot(ttmp, inj[:len(ttmp)])
    ax2[2,1].set_title("I_Inj")
    plt.show()
        
if __name__ == '__main__':
    run()
