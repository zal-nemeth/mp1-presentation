import pyvisa
import time
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
#matplotlib.use("Qt5Agg")
import numpy as np
import math
from datetime import datetime



def agilent_sweep(start=1500,
                  stop=1530,
                  resolution=1,
                  integration_time=0.001,
                  pwr=1.0,
                  twow=0,
                  sweep_s=10,
                  sn=-30):
    rm = pyvisa.ResourceManager()
    rm.list_resources()

    ls = rm.open_resource("TCPIP0::169.254.246.175::inst0::INSTR")
    #ls = rm.open_resource("TCPIP0::128.40.39.71::inst0::INSTR")
    pm = rm.open_resource("TCPIP0::169.254.192.57::inst0::INSTR")
    #agilent = rm.open_resource("GPIB0::20::INSTR")
    pm.write("*rst; status:preset; *cls")
    ls.write("*rst; status:preset; *cls")
    #start_w=1540
    #stop_w=1580
    #res=10
    #in_t=1
    pm.timeout=25000
    ls.timeout=25000
    ls.write(":SOUR0:WAV:SWE:MODE CONT")
    pwr=pwr*1000
    ls.write(":SOUR0:WAV:SWE:SPE %d nm/s"%sweep_s)
    ls.write(":SOUR0:WAV:SWE:STAR %d nm" %start)
    ls.write(":SOUR0:WAV:SWE:STOP %d nm" %stop)
    ls.write(":SOUR0:WAV:SWE:STEP %f pm" %resolution)
    ls.write(":SOUR0:POW %d UW"%pwr)
    number_data_points = abs(int(math.floor( (stop-start)/(resolution/1000.0)))) + 1
    if twow==1:
        ls.write(":SOUR0:WAV:SWE:REP twow")
        ls.write(":sour0:wav:swe:cycl 2")
        number_data_points = abs(int(math.floor(2* (stop-start)/(resolution/1000.0)))) + 1
    mode=ls.query("SOUR0:WAV:SWE:MODE?") 
    print("the mode is set to %s"%mode)
    mode=ls.query("SOUR0:WAV:SWE:REP?") 
    print("the mode is set to %s"%mode)    
    # calculate numebr of datapoints
    
    print(number_data_points)
    
    
    
#    ls.write(":SOUR0:WAV:SWE:CYCL 2")
    ls.write(":SOUR0:WAV:SWE:LLOG 1")
    ls.write(":SOUR0:OUTP1:STAT OFF")
    
    ls.write(":TRIG0:INP IGNORE")
    ls.write(":TRIG0:OUTP STFINISHED")
    
    pm.write("TRIG1:INP SMEASURE")
    pm.write("TRIG1:OUTP DISAB")
    
    #pm.write("TRIG1:CONF LOOP")
    print(pm.query("TRIG1:CONF?"))

    
    pm.write("SENS0:POW:RANG:AUTO 0")
    # check the setting
    test=ls.query(":SOUR:WAV:SWE:CHEC?")
    ls.write("OUTP0:STAT 1")
    print("test gave %s"%test)
    pm.write("SENS1:POW:UNIT 0")
    pm.write("SENS1:POW:RANG %d dBm"%sn)
    pm.write("SENS1:POW:WAV 1550nm")
    #averaging_time = pm.query("SENS1:POW:ATIM?")
    #pm.write("SENS1:POW:ATIM 50us")
    rang=pm.query("SENS1:POW:RANG?")
    print("the range is set to %s"%rang)
    
    pm.write("SENS1:FUNC:PAR:LOGG %d,%fms" %(number_data_points,integration_time))
    #averaging_time = pm.query("SENS1:POW:ATIM?")
    #print("the averaging time is set to %s"%averaging_time)    
    
    
    pm.write("SENS1:FUNC:STAT LOGGING,START")
    #time.sleep(1)#alfonso
    ls.write(":SOUR0:WAV:SWE:STATE START")
    
    wai=1
    print("sweeping")
    while wai==1:
        wai=int(ls.query(":SOUR0:WAV:SWE:STATE?"))
        #print(wai)
        #print(".....")
        if wai==0:
            break
    print("sweep is over") 
    ls.write("*opc?")

    
    out= pm.query_binary_values("SENS1:FUNCTION:RES?")
    powers=out
    print("powers size is %r"%(len(powers)))
    
    
    # reading the laser
    #time.sleep(0.5)#alfonso
    ls.write(":SOUR0:READ:DATA? LLOG")
    wavelengthss=ls.query_binary_values(":SOUR0:READ:DATA? LLOG", datatype='d', is_big_endian=False)

 

    print("wavelenght size is %r"%(len(wavelengthss)))

    ls.write("OUTP0:STAT 0")
    ls.close()
    pm.close()
    return wavelengthss, powers

if __name__ == "__main__":
    
    w,p=agilent_sweep(start=1550, stop=1560, resolution=0.1,pwr=15.849,twow=0, sweep_s=10, sn=-20)
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #np.savetxt(time_now+'cst_cb_50nm_4000um_0.4_wg_wg.txt', [w,p])
    np.savetxt(time_now+'mpw2_no600_f1_0_t1_0_wgw_1', [w,p])
    #np.savetxt(time_now+'LXT_A2_disk_1_0_0_4_10Vpp_5Vdc', [w,p])
    time.sleep(0.5)
    plt.plot(w,p)
    plt.savefig(time_now+'.svg')
    #'gr_cb_100nm_4000um_0.8_wg_wg.txt
   
    plt.show()
    '''
    pws=[]
    pw=np.linspace(0.1,15.0,30)
    pw=pw[::-1]
    #sp=np.linspace(1,199.0,50)
    sp=[0.5,1.0,2.0,5.0,10.0,20.0,40.0,50.0,80.0,100.0, 150.0, 160.0,200.0]
    #sp=np.logspace(np.log10(1),np.log10(199),50)
    print(sp)
    #print(sp)
    for x in pw:
        avers=[]
        for y in sp:
            xx=np.around(x,1)
            yy=np.around(y,1)
            if xx<=1.0:
                s=-20
            elif xx>1.0:
                s=-10
            w,p=agilent_sweep(start=1544, stop=1554, resolution=0.2,pwr=xx,twow=0, sweep_s=yy, sn=s)
            print(xx,yy)
            time.sleep(0.5)
            #plt.plot(w,p)
            #plt.show()
            aver=abs(np.around(np.average(p),2))
            pws.append(aver)
            np.savetxt('08_08_2022/ring27_%i_%i_%i.txt'%(xx*10,yy*10,abs(aver*100)), [w,p])
            print('the average power is %r'%aver)
            avers.append(aver)
        #plt.figure(2)
        #plt.plot(avers)
        #plt.show()
        #input("Press Enter to continue...") 
    #plt.show()    
    np.savetxt('08_08_2022/power.txt', pws)
    '''
    
    