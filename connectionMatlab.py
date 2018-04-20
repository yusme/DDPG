# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:10:12 2018

@author: yeimy_p
"""

import matlab.engine
import matplotlib.pyplot as plt
import connectionMatlab1 as CM


class ConnectionMatlab:

    def __init__(self):
        self.data = []
        self.ResultSimulation = {}

    def plotResults(self, x, y):
        plt.plot(x, y)
        plt.show();

    def Openconnection(self):
        eng = matlab.engine.start_matlab()
        ret = eng.sqrt(4.0)

        if (ret == 2.0):
            print('Run connection---Done!', ret)
            self.eng = eng;

    def LoadInputData(self):
        eng = self.eng
        eng.eval("load('input.mat')", nargout=0);
        var = eng.workspace['var']

        print ("\n")
        print("var:__", var)
        print('setInputData---Done!')






        # simulation 1

    def runSimulation(self, model, times):

        print("runSimulation");

        eng = self.eng

        eng.eval("sim('" + model + "')");

        # set Parameter für simulation
        eng.eval("set_param('" + model + "','StartTime','0','StopTime','80')", nargout=0);

        StartTime = times["StartTime"];
        StopTime = times["StopTime"];

        for n in range(0, 2):
            cad = cad = "set_param('" + model + "','StartTime','" + str(StartTime[n]) + "', 'StopTime' ,'" + str(
                StopTime[n]) + "')";

            print ("--", cad)

            # set Parameter für simulation
            eng.eval(cad, nargout=0)

            print ("'StartTime:'", eng.get_param(model, 'StartTime'))
            print ("'StopTime:'", eng.get_param(model, 'StopTime'))

            eng.eval("sim('" + model + "')");
            var = eng.workspace['ScopeData']
            # x = var['time']
            # y = var['signals']['values']


            # print ('ScopeData: ', var)


            # algorithmus
            # self.plotResults(x,y)
            print ("\n ENDE")






            # simulation 2  with input data

    def runSimulationInput(self, model):
        eng = self.eng
        eng.eval("sim('" + model + "')");
        var = eng.workspace['ScopeData']
        x = var['time']
        y = var['signals']['values']
        self.plotResults(x, y)




def main():
    model = 'simu'
    # result="ScopeData"


    # var={};
    times = {};
    times["StartTime"] = [0.0, 10.0, 30.0, 80.0, 110.0];
    times["StopTime"] = [10.0, 30.0, 80.0, 110.0, 250.0];

    print ("\n times ", times["StartTime"], " *** ", times["StopTime"])
    connection = CM.ConnectionMatlab()
    connection.Openconnection();
    # connection.runSimulation(model,times);

    print ("\n End first simulation \n\n")



    model = 'data_import'
    connection.LoadInputData();
    connection.runSimulationInput(model);

    print ("\n End second simulation \n\n")

    times["StartTime"] = [0.0, 0.0];
    times["StopTime"] = [0.06, 0.09]

    model = 'sldemo_h'
    connection.runSimulation(model, times);

    print ("\n End third simulation \n\n")
    # var={};
    # signals={};
    # signals["values"] = [0, 0, 2, 2, 2, 3, 3, 3];
    # signals["dimensions"] = 1;


# if __name__ == '__main__':
main()