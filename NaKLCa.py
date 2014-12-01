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
        self.Tfinal = 2500.0
        # output time steps
        self.dt = 0.01

        #Injected current file. Also need to specify time-step between
        #data points in file. Will linearly interpolate for intermediate
        #points
        self.inj = sp.loadtxt('current_sq.txt')
        self.injdt = 50.0
        #inj = sp.loadtxt('current_l63.txt')
        #injdt = 0.01

        self.gNa = 120.0 # nS
        #gNa = 0.0
        self.gK = 20.0 
        self.gL = .30 
        self.gSK = 35.0
        self.gCaL = 0.015
        self.phi = 0.0001 # uM*cm^2/uA/ms
        #gCaL = 0.005 # mS/cm^2/uM(micromolar) - This is different because uses GHK form of current
        #gSK = 9.1 

#        self.gNa = 1.20 # nS
#        #gNa = 0.0
#        self.gK = .20 
#        self.gL = .0030 
#        self.gSK = 0.500
#        self.gCaL = 0.00015
#        self.phi = 0.01 # uM*cm^2/uA/ms


        self.ENa = 50.0 #mV
        self.EK = -77.0 #mV
        self.EL = -80.4 #mV

        # The form of the sigmoid funcions is f(V) =
        # 1/(1+exp((V-theta)/sigma)) this is =
        # 0.5(1+tanh((V-th)/(-2*sig)));

        # In BioCyb V_a = th_a; dV_a = -2*sig_a

        self.th_m = -40.0 #mV
        self.th_h = -60.0 #mV
        self.th_n = -55.0 #mV
        self.th_s = -40.2 #mV
        self.sig_m = -7.5 #mV
        self.sig_h = 7.5 #mV
        self.sig_n = -15.0 #mV
        self.sig_s = -8.0 #mV

        self.t0_m = 0.1 #ms
        self.t0_h = 1.0 #ms
        self.t0_n = 1.0 #ms
        self.t0_s = 0.1 #ms

        self.t1_m = 0.4 #ms
        self.t1_h = 7.0 
        self.t1_n = 5.0
        self.t1_s = 0.2 # s-variable was not originally included in NaKL (came from Arij). Constant tau_s = t0_s seems as valid as anything else


        self.CaExt = 2500.0 # uM
        self.k_s = 2.3 # uM


        self.kCa = 0.01 # 1/ms
#        self.kCa = 0.05 
        self.bCa = 0.2 # uM
        self.C = 1.0 # uF
        #self.C = 0.01 # uF

        self.gSD = 1.0 #Conductances between Soma/Dendrite


        #PROBABLY NOT for estimation
        self.Isa = 10.0 # Scaling for Iinj: Iinj/Isa should be in [nA]
        self.f = 1.0 #unitless. Fraction of Free/Total cytosolic Ca^2+. Degenerate with params kCa, bCa.
        self.p = 2.0 #unitless. Exponent of hill fcn. Hardcoded in Arij model
        self.T = 290.0 #K . Temperature, Probably doesn't need to be estimated       
        


    def I_Na(self,V,m,h):   return self.gNa * m**3 * h * (self.ENa-V)

    def I_K(self,V,n):      return self.gK * n**4 * (self.EK-V)

    #note, this is leak current, not L-type Ca
    def I_L(self, V):        return self.gL * (self.EL-V)

    # This is SK current from Arij's model. I *think* SK is a specific
    # class of Ca-gated K currents
    def I_SK(self, V, Ca):
        kinf = Ca**self.p/(Ca**self.p + self.k_s**self.p)
        return self.gSK * kinf * (self.EK-V)

    def I_CaL(self, V, Ca,s):
        #23.209 is 2*F/R in K/mV
        return (self.gCaL * V * s**2 
                *(Ca- self.CaExt*sp.exp(-23.209*V/self.T))/(sp.exp(-23.209*V/self.T)-1))
        #    return gCaL * V * s**2 * (CaExt)/(1-sp.exp(23.209*V/T))

    def gating_inf(self, V,theta,sigma): 
    # Using exponential form for straightforward comparison with other
        # neuron models/parameters
        return 1.0/(1+sp.exp((V-theta)/sigma))

    def tau(self, V,t0,t1,theta,sigma):
        # I am using the exponetial form of x_inf(V), so there's a
        # relative factor of (-2) in the sigma.
        return t0 + t1*(1.0-sp.tanh((V-theta)/(-2.0*sigma))**2)    
    
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
                              + self.Iinj(t,current,tstep)/self.Isa 
                              + self.I_L(Vs) + self.gSD*(Vd-Vs)) 

        dVddt = 1.0/self.C * (self.I_SK(Vd,Ca) + self.I_CaL(Vd,Ca,s) 
                              + self.I_L(Vs) + self.gSD*(Vs-Vd))

        dCadt = self.f*(self.phi*self.I_CaL(Vd,Ca,s) + self.kCa*(self.bCa - Ca))     
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
        ax[0,0].set_title('Voltage Soma')
        ax[0,1].plot(times,sim[:,1])
        ax[0,1].set_title('Voltage Dend')
        ax[1,0].set_title('Calcium')
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
        ax2[0,0].set_title('I_Na (\mu A/cm^2)')
        ax2[0,1].plot(times,self.I_K(sim[:,0],sim[:,5]))
        ax2[0,1].set_title('I_K (\mu A/cm^2)')
        ax2[1,0].plot(times,self.I_L(sim[:,0]))
        ax2[1,0].set_title('I_L (\mu A/cm^2)')
        ax2[1,1].plot(times,self.I_SK(sim[:,1], sim[:,2]))
        ax2[1,1].set_title("I_SK (\mu A/cm^2)")
        ax2[2,0].plot(times,self.I_CaL(sim[:,1],sim[:,2],sim[:,6]))
        ax2[2,0].set_title("I_CaL (\mu A/cm^2)")             
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
    
    
