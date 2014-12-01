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

# The I_SK is taken from Arij's model, with some parameter values
# taken from Nogaret

class NaKLCa_Neuron:
    def __init__(self):
        
        # Initial Conditions
        self.init = [-70.0, -70.0, 1.0, 0.1, 0.7, 0.1, 0.1]
        # Total Integration time and 
        self.Tfinal = 2400.0
        # output time steps
        self.dt = 0.01

        #Injected current file. Also need to specify time-step between
        #data points in file. Will linearly interpolate for intermediate
        #points
        self.inj = sp.loadtxt('current_sq_nA.txt')
        #inj = sp.loadtxt('current_l63.txt')
        self.injdt = 0.01



        self.gNa = 1.20 # nS
        self.gK = 0.20 
        self.gL = 0.003 
        self.gKCa = 0.35
        self.gCaL = 0.00015  # uS/uM
        self.phi = 0.01 # uM/ms/nA

        self.ENa = 50.0 #mV
        self.EK = -77.0 #mV
        self.EL = -80.4 #mV

        self.th_m = -40.0 #mV
        self.th_h = -60.0 #mV
        self.th_n = -55.0 #mV
        self.th_s = -40.2 #mV
        self.sig_m = 15.0 #mV
        self.sig_h = -15.0 #mV
        self.sig_n = 30.0 #mV
        self.sig_s = 16.0 #mV

        self.t0_m = 0.1 #ms
        self.t0_h = 1.0 #ms
        self.t0_n = 1.0 #ms
        self.t0_s = 0.1 #ms

        self.t1_m = 0.4 #ms
        self.t1_h = 7.0 
        self.t1_n = 5.0
        self.t1_s = 0.2 

        self.CaExt = 2500.0 # uM
        self.k_s = 2.5 # uM


        self.t_Ca = 50.0 # ms
        self.Ca0 = 0.2 # uM
        self.C = 0.01 # uF


        self.gSD = 0.01 #Conductances between Soma/Dendrite


        #PROBABLY NOT for estimation
        self.Isa = 1.0 # Scaling for Iinj: Iinj/Isa should be in [nA]

        self.p = 2.0 #unitless. Exponent of hill fcn. Hardcoded in Arij model
        self.T = 290.0 #K . Temperature, Probably doesn't need to be estimated       
      
    def I_Na(self,V,m,h):   return self.gNa * m**3 * h * (self.ENa-V)

    def I_K(self,V,n):      return self.gK * n**4 * (self.EK-V)

    #note, this is leak current, not L-type Ca
    def I_L(self, V):        return self.gL * (self.EL-V)

    # This is SK current from Arij's model. I *think* SK is a specific
    # class of Ca-gated K currents
    def I_KCa(self, V, Ca):
        kinf = Ca**self.p/(Ca**self.p + self.k_s**self.p)
        return self.gKCa * kinf * (self.EK-V)

    def I_CaL(self, V, Ca,s):
        #23.209 is 2*F/R in K/mV
        return (self.gCaL * V * s**2 
                *(Ca- self.CaExt*sp.exp(-23.209*V/self.T))/(sp.exp(-23.209*V/self.T)-1))
        #    return gCaL * V * s**2 * (CaExt)/(1-sp.exp(23.209*V/T))

    def gating_inf(self, V,theta,sigma): 
    # Using exponential form for straightforward comparison with other
    # neuron models/parameters
        #return 1.0/(1+sp.exp((V-theta)/sigma))
        return 0.5*(1.0+sp.tanh((V-theta)/sigma))

    def tau(self, V,t0,t1,theta,sigma):
        # I am using the exponetial form of x_inf(V), so there's a
        # relative factor of (-2) in the sigma.
        #return t0 + t1*(1.0-sp.tanh((V-theta)/(-2.0*sigma))**2)    
        return t0 + t1*(1.0-sp.tanh((V-theta)/(sigma))**2)    
    
    # Give this function an array of current values and the time step
    # between them, it will interpolate using a simple line between
    # neighboring time points
    # Current is an array of current data points, seperated by "step" amount of time
    def Iinj(self, t, Ivalue, step): 
        idx = int(t/step)
        if idx == t/step:
            return Ivalue[idx]
        else:
            slope = (Ivalue[idx+1]-Ivalue[idx])/step               
            return Ivalue[idx] + slope*(t-idx*step)    
    
    def eqns(self,x, t, current, tstep):
        Vs, Vd, Ca, m, h, n, s = x            

        dVsdt = 1.0/self.C * (self.I_Na(Vs,m,h) + self.I_K(Vs,n) 
                              + self.I_L(Vs) + self.gSD*(Vd-Vs)) 

        dVddt = 1.0/self.C * (self.I_KCa(Vd,Ca) + self.I_CaL(Vd,Ca,s) 
                              + self.Iinj(t,current,tstep)/self.Isa 
                              + self.I_L(Vs) + self.gSD*(Vs-Vd))

        dCadt = self.phi*self.I_CaL(Vd,Ca,s) + (self.Ca0 - Ca)/self.t_Ca
        dmdt = ((self.gating_inf(Vs,self.th_m,self.sig_m)-m)
                / self.tau(Vs,self.t0_m,self.t1_m,self.th_m,self.sig_m))
        dhdt = ((self.gating_inf(Vs,self.th_h,self.sig_h)-h)
                / self.tau(Vs,self.t0_h,self.t1_h,self.th_h,self.sig_h))
        dndt = ((self.gating_inf(Vs,self.th_n,self.sig_n)-n)
                / self.tau(Vs,self.t0_n,self.t1_n,self.th_n,self.sig_n))
        dsdt = ((self.gating_inf(Vd,self.th_s,self.sig_s)-s)
                / self.tau(Vd,self.t0_s,self.t1_s,self.th_s,self.sig_s))

        return dVsdt, dVddt, dCadt, dmdt, dhdt, dndt, dsdt
    

    def run(self):
    
        # Parameters passed are current data array, along with time step
        # between current data points
        self.times = sp.arange(0,self.Tfinal,self.dt)
        self.sim = odeint(self.eqns,self.init,self.times,(self.inj,self.injdt))
        sp.savetxt('simulation.txt',sp.column_stack((self.times,self.sim)))

    def plot(self):
        sim = self.sim
        times = self.times
        inj = self.inj
        injdt = self.injdt    

        fig, ax = plt.subplots(3,2,sharex=True)        
        ax[0,0].plot(times,sim[:,0])
        ax[0,0].set_title('Voltage Soma (mV)')
        ax[0,1].plot(times,sim[:,1])
        ax[0,1].set_title('Voltage Dend (mV)')
        ax[1,0].set_title('Internal Calcium ($\mu$M)')
        ax[1,0].plot(times,sim[:,2])
        ax[1,1].set_title('m (Na act)')
        ax[1,1].plot(times,sim[:,3])
        ax[2,0].set_title('h (Na inact)')
        ax[2,0].plot(times,sim[:,4])
#        ax[2,0].set_title('n (K act)')
        ax[2,1].plot(times,sim[:,6])
        ax[2,1].set_title('s (CaL act)')
        fig.tight_layout()

        fig2, ax2 = plt.subplots(3,2,sharex=True)
        ax2[0,0].plot(times,self.I_Na(sim[:,0],sim[:,3],sim[:,4]))
        ax2[0,0].set_title('$I_{Na}$ (nA)')
        ax2[0,1].plot(times,self.I_K(sim[:,0],sim[:,5]))
        ax2[0,1].set_title('$I_K$ (nA)')
        ax2[1,0].plot(times,self.I_L(sim[:,0]))
        ax2[1,0].set_title('$I_L$ (nA)')
        ax2[1,1].plot(times,self.I_KCa(sim[:,1], sim[:,2]))
        ax2[1,1].set_title("$I_{KCa}$ (nA)")
        ax2[2,0].plot(times,self.I_CaL(sim[:,1],sim[:,2],sim[:,6]))
        ax2[2,0].set_title("$I_{CaL}$ (nA)")             
        fig2.tight_layout()

        ttmp = sp.arange(0,self.Tfinal,self.injdt)
        ax2[2,1].plot(ttmp, self.inj[:len(ttmp)]/self.Isa)
        ax2[2,1].set_title("I_Inj")
#
#        fig3, ax3 = plt.subplots(2,1,sharex=True)
#
#        ax3[0].plot(times, sim[:,0])
#        ax3[0].set_title("Voltage (mV)")
#        ax3[1].plot(ttmp, self.inj[:len(ttmp)]/self.Isa)
#        ax3[1].set_title("Inj Current")
#        fig3.tight_layout()

        plt.show()
        
if __name__ == '__main__':

    neuron1 = NaKLCa_Neuron()
#    t0_sinit = neuron1.t0_s
#    t1_sinit = neuron1.t1_s
#    fig, ax = plt.subplots(6, sharex=True)
#    for i in range(5):
#        neuron1.t0_s = 5.0**(i-3)*t0_sinit
#        neuron1.t1_s = 5.0**(i-3)*t1_sinit
#        neuron1.run()
#        ax[i].plot(neuron1.times,neuron1.sim[:,0])
#        ax[i].set_title("Voltage (mV), with t0_s = {}".format(neuron1.t0_s))
#
#    ttmp = sp.arange(0,neuron1.Tfinal,neuron1.injdt)
#    ax[5].plot(ttmp, neuron1.inj[:len(ttmp)]/neuron1.Isa)
#    ax[5].set_title("Inj Current (\mu A)")
#    ax[5].set_xlabel("Time (ms)")
#    fig.tight_layout()
#    plt.show()


    neuron1.run()
    neuron1.plot()
    
    
