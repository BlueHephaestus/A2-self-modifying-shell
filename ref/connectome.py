# GoPiGo Connectome
# Written by Timothy Busbice, Gabriel Garrett, Geoffrey Churchill (c) 2014, in Python 2.7
# The GoPiGo Connectome uses a post_synaptic dictionary based on the C Elegans Connectome Model
# This application can be ran on the Raspberry Pi GoPiGo robot with a Sonar that represents Nose Touch when activated
# To run standalone without a GoPiGo robot, simply comment out the sections with Start and End comments 

# TIME STATE EXPERIMENTAL OPTIMIZATION
# The previous version had a logic error whereby if more than one neuron fired into the same neuron in the next time state,
# it would overwrite the contribution from the previous neuron. Thus, only one neuron could fire into the same neuron at any given time state.
# This version also explicitly lists all left and right muscles, so that during the muscle checks for the motor control function, instead of 
# iterating through each neuron, we now iterate only through the relevant muscle neurons.

# This logic error took it's place via doing neuron[nextstate] = neuron[thisState] + 2
# so if there were ever any neurons from this state that were going to influence the next state,
# this would not get counted because it would reassign it.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
# three levels of verbosity: [], -v, -vv
# 0: only print starting and exiting messages
# 1: print only when obstacles/food found
# 2: print speed, left and right every time
verbosity = parser.parse_args().verbose

from body import Body
body = Body()

import time
import copy
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from neurons import *

class Nematode:

    def __init__(self):
        # The post_synaptic dictionary contains the accumulated weighted values as the
        # connectome is executed
        self.post_synaptic = {}

        # todo make this self.curr and self.next
        self.thisState = 0
        self.nextState = 1

        # The Threshold is the maximum sccumulated value that must be exceeded before
        # the Neurite will fire
        self.threshold = 30

        self.time_delays = False
        # Accumulators are used to decide the value to send to the Left and Right motors
        # of the GoPiGo robot
        self.accumleft = 0
        self.accumright = 0

        # Used to remove from Axon firing since muscles cannot fire.
        self.muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

        self.mLeft =  ['MDL01', 'MDL02', 'MDL03', 'MDL04', 'MDL05', 'MDL06', 'MDL07', 'MDL08', 'MDL09', 'MDL10',
                  'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20',
                  'MDL21', 'MDL22', 'MDL23', 'MDL24', 'MVL01', 'MVL02', 'MVL03', 'MVL04', 'MVL05', 'MVL06',
                  'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16',
                  'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']

        self.mRight = ['MDR01', 'MDR02', 'MDR03', 'MDR04', 'MDR05', 'MDR06', 'MDR07', 'MDR08', 'MDR09', 'MDR10',
                  'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20',
                  'MDR21', 'MDR22', 'MDR23', 'MDR24', 'MVR01', 'MVR02', 'MVR03', 'MVR04', 'MVR05', 'MVR06',
                  'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16',
                  'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVR21', 'MVR22', 'MVR23']
        # Used to accumulate muscle weighted values in body muscles 07-23 = worm locomotion


        """This is the full C Elegans Connectome as expresed in the form of the Presynatptic
        neurite and the post_synaptic neurites.

        post_synaptic['ADAR'][nextState] = (2 + post_synaptic['ADAR'][thisState])
        arr=post_synaptic['AIBR'] potential optimization
        """
        # Create the dictionary
        self.create_post_synaptic()





    def create_post_synaptic(self):
        # The post_synaptic dictionary maintains the accumulated values for
        # each neuron and muscle. The Accumulated values are initialized to Zero
        self.post_synaptic['ADAL'] = [0, 0]
        self.post_synaptic['ADAR'] = [0, 0]
        self.post_synaptic['ADEL'] = [0, 0]
        self.post_synaptic['ADER'] = [0, 0]
        self.post_synaptic['ADFL'] = [0, 0]
        self.post_synaptic['ADFR'] = [0, 0]
        self.post_synaptic['ADLL'] = [0, 0]
        self.post_synaptic['ADLR'] = [0, 0]
        self.post_synaptic['AFDL'] = [0, 0]
        self.post_synaptic['AFDR'] = [0, 0]
        self.post_synaptic['AIAL'] = [0, 0]
        self.post_synaptic['AIAR'] = [0, 0]
        self.post_synaptic['AIBL'] = [0, 0]
        self.post_synaptic['AIBR'] = [0, 0]
        self.post_synaptic['AIML'] = [0, 0]
        self.post_synaptic['AIMR'] = [0, 0]
        self.post_synaptic['AINL'] = [0, 0]
        self.post_synaptic['AINR'] = [0, 0]
        self.post_synaptic['AIYL'] = [0, 0]
        self.post_synaptic['AIYR'] = [0, 0]
        self.post_synaptic['AIZL'] = [0, 0]
        self.post_synaptic['AIZR'] = [0, 0]
        self.post_synaptic['ALA'] = [0, 0]
        self.post_synaptic['ALML'] = [0, 0]
        self.post_synaptic['ALMR'] = [0, 0]
        self.post_synaptic['ALNL'] = [0, 0]
        self.post_synaptic['ALNR'] = [0, 0]
        self.post_synaptic['AQR'] = [0, 0]
        self.post_synaptic['AS1'] = [0, 0]
        self.post_synaptic['AS10'] = [0, 0]
        self.post_synaptic['AS11'] = [0, 0]
        self.post_synaptic['AS2'] = [0, 0]
        self.post_synaptic['AS3'] = [0, 0]
        self.post_synaptic['AS4'] = [0, 0]
        self.post_synaptic['AS5'] = [0, 0]
        self.post_synaptic['AS6'] = [0, 0]
        self.post_synaptic['AS7'] = [0, 0]
        self.post_synaptic['AS8'] = [0, 0]
        self.post_synaptic['AS9'] = [0, 0]
        self.post_synaptic['ASEL'] = [0, 0]
        self.post_synaptic['ASER'] = [0, 0]
        self.post_synaptic['ASGL'] = [0, 0]
        self.post_synaptic['ASGR'] = [0, 0]
        self.post_synaptic['ASHL'] = [0, 0]
        self.post_synaptic['ASHR'] = [0, 0]
        self.post_synaptic['ASIL'] = [0, 0]
        self.post_synaptic['ASIR'] = [0, 0]
        self.post_synaptic['ASJL'] = [0, 0]
        self.post_synaptic['ASJR'] = [0, 0]
        self.post_synaptic['ASKL'] = [0, 0]
        self.post_synaptic['ASKR'] = [0, 0]
        self.post_synaptic['AUAL'] = [0, 0]
        self.post_synaptic['AUAR'] = [0, 0]
        self.post_synaptic['AVAL'] = [0, 0]
        self.post_synaptic['AVAR'] = [0, 0]
        self.post_synaptic['AVBL'] = [0, 0]
        self.post_synaptic['AVBR'] = [0, 0]
        self.post_synaptic['AVDL'] = [0, 0]
        self.post_synaptic['AVDR'] = [0, 0]
        self.post_synaptic['AVEL'] = [0, 0]
        self.post_synaptic['AVER'] = [0, 0]
        self.post_synaptic['AVFL'] = [0, 0]
        self.post_synaptic['AVFR'] = [0, 0]
        self.post_synaptic['AVG'] = [0, 0]
        self.post_synaptic['AVHL'] = [0, 0]
        self.post_synaptic['AVHR'] = [0, 0]
        self.post_synaptic['AVJL'] = [0, 0]
        self.post_synaptic['AVJR'] = [0, 0]
        self.post_synaptic['AVKL'] = [0, 0]
        self.post_synaptic['AVKR'] = [0, 0]
        self.post_synaptic['AVL'] = [0, 0]
        self.post_synaptic['AVM'] = [0, 0]
        self.post_synaptic['AWAL'] = [0, 0]
        self.post_synaptic['AWAR'] = [0, 0]
        self.post_synaptic['AWBL'] = [0, 0]
        self.post_synaptic['AWBR'] = [0, 0]
        self.post_synaptic['AWCL'] = [0, 0]
        self.post_synaptic['AWCR'] = [0, 0]
        self.post_synaptic['BAGL'] = [0, 0]
        self.post_synaptic['BAGR'] = [0, 0]
        self.post_synaptic['BDUL'] = [0, 0]
        self.post_synaptic['BDUR'] = [0, 0]
        self.post_synaptic['CEPDL'] = [0, 0]
        self.post_synaptic['CEPDR'] = [0, 0]
        self.post_synaptic['CEPVL'] = [0, 0]
        self.post_synaptic['CEPVR'] = [0, 0]
        self.post_synaptic['DA1'] = [0, 0]
        self.post_synaptic['DA2'] = [0, 0]
        self.post_synaptic['DA3'] = [0, 0]
        self.post_synaptic['DA4'] = [0, 0]
        self.post_synaptic['DA5'] = [0, 0]
        self.post_synaptic['DA6'] = [0, 0]
        self.post_synaptic['DA7'] = [0, 0]
        self.post_synaptic['DA8'] = [0, 0]
        self.post_synaptic['DA9'] = [0, 0]
        self.post_synaptic['DB1'] = [0, 0]
        self.post_synaptic['DB2'] = [0, 0]
        self.post_synaptic['DB3'] = [0, 0]
        self.post_synaptic['DB4'] = [0, 0]
        self.post_synaptic['DB5'] = [0, 0]
        self.post_synaptic['DB6'] = [0, 0]
        self.post_synaptic['DB7'] = [0, 0]
        self.post_synaptic['DD1'] = [0, 0]
        self.post_synaptic['DD2'] = [0, 0]
        self.post_synaptic['DD3'] = [0, 0]
        self.post_synaptic['DD4'] = [0, 0]
        self.post_synaptic['DD5'] = [0, 0]
        self.post_synaptic['DD6'] = [0, 0]
        self.post_synaptic['DVA'] = [0, 0]
        self.post_synaptic['DVB'] = [0, 0]
        self.post_synaptic['DVC'] = [0, 0]
        self.post_synaptic['FLPL'] = [0, 0]
        self.post_synaptic['FLPR'] = [0, 0]
        self.post_synaptic['HSNL'] = [0, 0]
        self.post_synaptic['HSNR'] = [0, 0]
        self.post_synaptic['I1L'] = [0, 0]
        self.post_synaptic['I1R'] = [0, 0]
        self.post_synaptic['I2L'] = [0, 0]
        self.post_synaptic['I2R'] = [0, 0]
        self.post_synaptic['I3'] = [0, 0]
        self.post_synaptic['I4'] = [0, 0]
        self.post_synaptic['I5'] = [0, 0]
        self.post_synaptic['I6'] = [0, 0]
        self.post_synaptic['IL1DL'] = [0, 0]
        self.post_synaptic['IL1DR'] = [0, 0]
        self.post_synaptic['IL1L'] = [0, 0]
        self.post_synaptic['IL1R'] = [0, 0]
        self.post_synaptic['IL1VL'] = [0, 0]
        self.post_synaptic['IL1VR'] = [0, 0]
        self.post_synaptic['IL2L'] = [0, 0]
        self.post_synaptic['IL2R'] = [0, 0]
        self.post_synaptic['IL2DL'] = [0, 0]
        self.post_synaptic['IL2DR'] = [0, 0]
        self.post_synaptic['IL2VL'] = [0, 0]
        self.post_synaptic['IL2VR'] = [0, 0]
        self.post_synaptic['LUAL'] = [0, 0]
        self.post_synaptic['LUAR'] = [0, 0]
        self.post_synaptic['M1'] = [0, 0]
        self.post_synaptic['M2L'] = [0, 0]
        self.post_synaptic['M2R'] = [0, 0]
        self.post_synaptic['M3L'] = [0, 0]
        self.post_synaptic['M3R'] = [0, 0]
        self.post_synaptic['M4'] = [0, 0]
        self.post_synaptic['M5'] = [0, 0]
        self.post_synaptic['MANAL'] = [0, 0]
        self.post_synaptic['MCL'] = [0, 0]
        self.post_synaptic['MCR'] = [0, 0]
        self.post_synaptic['MDL01'] = [0, 0]
        self.post_synaptic['MDL02'] = [0, 0]
        self.post_synaptic['MDL03'] = [0, 0]
        self.post_synaptic['MDL04'] = [0, 0]
        self.post_synaptic['MDL05'] = [0, 0]
        self.post_synaptic['MDL06'] = [0, 0]
        self.post_synaptic['MDL07'] = [0, 0]
        self.post_synaptic['MDL08'] = [0, 0]
        self.post_synaptic['MDL09'] = [0, 0]
        self.post_synaptic['MDL10'] = [0, 0]
        self.post_synaptic['MDL11'] = [0, 0]
        self.post_synaptic['MDL12'] = [0, 0]
        self.post_synaptic['MDL13'] = [0, 0]
        self.post_synaptic['MDL14'] = [0, 0]
        self.post_synaptic['MDL15'] = [0, 0]
        self.post_synaptic['MDL16'] = [0, 0]
        self.post_synaptic['MDL17'] = [0, 0]
        self.post_synaptic['MDL18'] = [0, 0]
        self.post_synaptic['MDL19'] = [0, 0]
        self.post_synaptic['MDL20'] = [0, 0]
        self.post_synaptic['MDL21'] = [0, 0]
        self.post_synaptic['MDL22'] = [0, 0]
        self.post_synaptic['MDL23'] = [0, 0]
        self.post_synaptic['MDL24'] = [0, 0]
        self.post_synaptic['MDR01'] = [0, 0]
        self.post_synaptic['MDR02'] = [0, 0]
        self.post_synaptic['MDR03'] = [0, 0]
        self.post_synaptic['MDR04'] = [0, 0]
        self.post_synaptic['MDR05'] = [0, 0]
        self.post_synaptic['MDR06'] = [0, 0]
        self.post_synaptic['MDR07'] = [0, 0]
        self.post_synaptic['MDR08'] = [0, 0]
        self.post_synaptic['MDR09'] = [0, 0]
        self.post_synaptic['MDR10'] = [0, 0]
        self.post_synaptic['MDR11'] = [0, 0]
        self.post_synaptic['MDR12'] = [0, 0]
        self.post_synaptic['MDR13'] = [0, 0]
        self.post_synaptic['MDR14'] = [0, 0]
        self.post_synaptic['MDR15'] = [0, 0]
        self.post_synaptic['MDR16'] = [0, 0]
        self.post_synaptic['MDR17'] = [0, 0]
        self.post_synaptic['MDR18'] = [0, 0]
        self.post_synaptic['MDR19'] = [0, 0]
        self.post_synaptic['MDR20'] = [0, 0]
        self.post_synaptic['MDR21'] = [0, 0]
        self.post_synaptic['MDR22'] = [0, 0]
        self.post_synaptic['MDR23'] = [0, 0]
        self.post_synaptic['MDR24'] = [0, 0]
        self.post_synaptic['MI'] = [0, 0]
        self.post_synaptic['MVL01'] = [0, 0]
        self.post_synaptic['MVL02'] = [0, 0]
        self.post_synaptic['MVL03'] = [0, 0]
        self.post_synaptic['MVL04'] = [0, 0]
        self.post_synaptic['MVL05'] = [0, 0]
        self.post_synaptic['MVL06'] = [0, 0]
        self.post_synaptic['MVL07'] = [0, 0]
        self.post_synaptic['MVL08'] = [0, 0]
        self.post_synaptic['MVL09'] = [0, 0]
        self.post_synaptic['MVL10'] = [0, 0]
        self.post_synaptic['MVL11'] = [0, 0]
        self.post_synaptic['MVL12'] = [0, 0]
        self.post_synaptic['MVL13'] = [0, 0]
        self.post_synaptic['MVL14'] = [0, 0]
        self.post_synaptic['MVL15'] = [0, 0]
        self.post_synaptic['MVL16'] = [0, 0]
        self.post_synaptic['MVL17'] = [0, 0]
        self.post_synaptic['MVL18'] = [0, 0]
        self.post_synaptic['MVL19'] = [0, 0]
        self.post_synaptic['MVL20'] = [0, 0]
        self.post_synaptic['MVL21'] = [0, 0]
        self.post_synaptic['MVL22'] = [0, 0]
        self.post_synaptic['MVL23'] = [0, 0]
        self.post_synaptic['MVR01'] = [0, 0]
        self.post_synaptic['MVR02'] = [0, 0]
        self.post_synaptic['MVR03'] = [0, 0]
        self.post_synaptic['MVR04'] = [0, 0]
        self.post_synaptic['MVR05'] = [0, 0]
        self.post_synaptic['MVR06'] = [0, 0]
        self.post_synaptic['MVR07'] = [0, 0]
        self.post_synaptic['MVR08'] = [0, 0]
        self.post_synaptic['MVR09'] = [0, 0]
        self.post_synaptic['MVR10'] = [0, 0]
        self.post_synaptic['MVR11'] = [0, 0]
        self.post_synaptic['MVR12'] = [0, 0]
        self.post_synaptic['MVR13'] = [0, 0]
        self.post_synaptic['MVR14'] = [0, 0]
        self.post_synaptic['MVR15'] = [0, 0]
        self.post_synaptic['MVR16'] = [0, 0]
        self.post_synaptic['MVR17'] = [0, 0]
        self.post_synaptic['MVR18'] = [0, 0]
        self.post_synaptic['MVR19'] = [0, 0]
        self.post_synaptic['MVR20'] = [0, 0]
        self.post_synaptic['MVR21'] = [0, 0]
        self.post_synaptic['MVR22'] = [0, 0]
        self.post_synaptic['MVR23'] = [0, 0]
        self.post_synaptic['MVULVA'] = [0, 0]
        self.post_synaptic['NSML'] = [0, 0]
        self.post_synaptic['NSMR'] = [0, 0]
        self.post_synaptic['OLLL'] = [0, 0]
        self.post_synaptic['OLLR'] = [0, 0]
        self.post_synaptic['OLQDL'] = [0, 0]
        self.post_synaptic['OLQDR'] = [0, 0]
        self.post_synaptic['OLQVL'] = [0, 0]
        self.post_synaptic['OLQVR'] = [0, 0]
        self.post_synaptic['PDA'] = [0, 0]
        self.post_synaptic['PDB'] = [0, 0]
        self.post_synaptic['PDEL'] = [0, 0]
        self.post_synaptic['PDER'] = [0, 0]
        self.post_synaptic['PHAL'] = [0, 0]
        self.post_synaptic['PHAR'] = [0, 0]
        self.post_synaptic['PHBL'] = [0, 0]
        self.post_synaptic['PHBR'] = [0, 0]
        self.post_synaptic['PHCL'] = [0, 0]
        self.post_synaptic['PHCR'] = [0, 0]
        self.post_synaptic['PLML'] = [0, 0]
        self.post_synaptic['PLMR'] = [0, 0]
        self.post_synaptic['PLNL'] = [0, 0]
        self.post_synaptic['PLNR'] = [0, 0]
        self.post_synaptic['PQR'] = [0, 0]
        self.post_synaptic['PVCL'] = [0, 0]
        self.post_synaptic['PVCR'] = [0, 0]
        self.post_synaptic['PVDL'] = [0, 0]
        self.post_synaptic['PVDR'] = [0, 0]
        self.post_synaptic['PVM'] = [0, 0]
        self.post_synaptic['PVNL'] = [0, 0]
        self.post_synaptic['PVNR'] = [0, 0]
        self.post_synaptic['PVPL'] = [0, 0]
        self.post_synaptic['PVPR'] = [0, 0]
        self.post_synaptic['PVQL'] = [0, 0]
        self.post_synaptic['PVQR'] = [0, 0]
        self.post_synaptic['PVR'] = [0, 0]
        self.post_synaptic['PVT'] = [0, 0]
        self.post_synaptic['PVWL'] = [0, 0]
        self.post_synaptic['PVWR'] = [0, 0]
        self.post_synaptic['RIAL'] = [0, 0]
        self.post_synaptic['RIAR'] = [0, 0]
        self.post_synaptic['RIBL'] = [0, 0]
        self.post_synaptic['RIBR'] = [0, 0]
        self.post_synaptic['RICL'] = [0, 0]
        self.post_synaptic['RICR'] = [0, 0]
        self.post_synaptic['RID'] = [0, 0]
        self.post_synaptic['RIFL'] = [0, 0]
        self.post_synaptic['RIFR'] = [0, 0]
        self.post_synaptic['RIGL'] = [0, 0]
        self.post_synaptic['RIGR'] = [0, 0]
        self.post_synaptic['RIH'] = [0, 0]
        self.post_synaptic['RIML'] = [0, 0]
        self.post_synaptic['RIMR'] = [0, 0]
        self.post_synaptic['RIPL'] = [0, 0]
        self.post_synaptic['RIPR'] = [0, 0]
        self.post_synaptic['RIR'] = [0, 0]
        self.post_synaptic['RIS'] = [0, 0]
        self.post_synaptic['RIVL'] = [0, 0]
        self.post_synaptic['RIVR'] = [0, 0]
        self.post_synaptic['RMDDL'] = [0, 0]
        self.post_synaptic['RMDDR'] = [0, 0]
        self.post_synaptic['RMDL'] = [0, 0]
        self.post_synaptic['RMDR'] = [0, 0]
        self.post_synaptic['RMDVL'] = [0, 0]
        self.post_synaptic['RMDVR'] = [0, 0]
        self.post_synaptic['RMED'] = [0, 0]
        self.post_synaptic['RMEL'] = [0, 0]
        self.post_synaptic['RMER'] = [0, 0]
        self.post_synaptic['RMEV'] = [0, 0]
        self.post_synaptic['RMFL'] = [0, 0]
        self.post_synaptic['RMFR'] = [0, 0]
        self.post_synaptic['RMGL'] = [0, 0]
        self.post_synaptic['RMGR'] = [0, 0]
        self.post_synaptic['RMHL'] = [0, 0]
        self.post_synaptic['RMHR'] = [0, 0]
        self.post_synaptic['SAADL'] = [0, 0]
        self.post_synaptic['SAADR'] = [0, 0]
        self.post_synaptic['SAAVL'] = [0, 0]
        self.post_synaptic['SAAVR'] = [0, 0]
        self.post_synaptic['SABD'] = [0, 0]
        self.post_synaptic['SABVL'] = [0, 0]
        self.post_synaptic['SABVR'] = [0, 0]
        self.post_synaptic['SDQL'] = [0, 0]
        self.post_synaptic['SDQR'] = [0, 0]
        self.post_synaptic['SIADL'] = [0, 0]
        self.post_synaptic['SIADR'] = [0, 0]
        self.post_synaptic['SIAVL'] = [0, 0]
        self.post_synaptic['SIAVR'] = [0, 0]
        self.post_synaptic['SIBDL'] = [0, 0]
        self.post_synaptic['SIBDR'] = [0, 0]
        self.post_synaptic['SIBVL'] = [0, 0]
        self.post_synaptic['SIBVR'] = [0, 0]
        self.post_synaptic['SMBDL'] = [0, 0]
        self.post_synaptic['SMBDR'] = [0, 0]
        self.post_synaptic['SMBVL'] = [0, 0]
        self.post_synaptic['SMBVR'] = [0, 0]
        self.post_synaptic['SMDDL'] = [0, 0]
        self.post_synaptic['SMDDR'] = [0, 0]
        self.post_synaptic['SMDVL'] = [0, 0]
        self.post_synaptic['SMDVR'] = [0, 0]
        self.post_synaptic['URADL'] = [0, 0]
        self.post_synaptic['URADR'] = [0, 0]
        self.post_synaptic['URAVL'] = [0, 0]
        self.post_synaptic['URAVR'] = [0, 0]
        self.post_synaptic['URBL'] = [0, 0]
        self.post_synaptic['URBR'] = [0, 0]
        self.post_synaptic['URXL'] = [0, 0]
        self.post_synaptic['URXR'] = [0, 0]
        self.post_synaptic['URYDL'] = [0, 0]
        self.post_synaptic['URYDR'] = [0, 0]
        self.post_synaptic['URYVL'] = [0, 0]
        self.post_synaptic['URYVR'] = [0, 0]
        self.post_synaptic['VA1'] = [0, 0]
        self.post_synaptic['VA10'] = [0, 0]
        self.post_synaptic['VA11'] = [0, 0]
        self.post_synaptic['VA12'] = [0, 0]
        self.post_synaptic['VA2'] = [0, 0]
        self.post_synaptic['VA3'] = [0, 0]
        self.post_synaptic['VA4'] = [0, 0]
        self.post_synaptic['VA5'] = [0, 0]
        self.post_synaptic['VA6'] = [0, 0]
        self.post_synaptic['VA7'] = [0, 0]
        self.post_synaptic['VA8'] = [0, 0]
        self.post_synaptic['VA9'] = [0, 0]
        self.post_synaptic['VB1'] = [0, 0]
        self.post_synaptic['VB10'] = [0, 0]
        self.post_synaptic['VB11'] = [0, 0]
        self.post_synaptic['VB2'] = [0, 0]
        self.post_synaptic['VB3'] = [0, 0]
        self.post_synaptic['VB4'] = [0, 0]
        self.post_synaptic['VB5'] = [0, 0]
        self.post_synaptic['VB6'] = [0, 0]
        self.post_synaptic['VB7'] = [0, 0]
        self.post_synaptic['VB8'] = [0, 0]
        self.post_synaptic['VB9'] = [0, 0]
        self.post_synaptic['VC1'] = [0, 0]
        self.post_synaptic['VC2'] = [0, 0]
        self.post_synaptic['VC3'] = [0, 0]
        self.post_synaptic['VC4'] = [0, 0]
        self.post_synaptic['VC5'] = [0, 0]
        self.post_synaptic['VC6'] = [0, 0]
        self.post_synaptic['VD1'] = [0, 0]
        self.post_synaptic['VD10'] = [0, 0]
        self.post_synaptic['VD11'] = [0, 0]
        self.post_synaptic['VD12'] = [0, 0]
        self.post_synaptic['VD13'] = [0, 0]
        self.post_synaptic['VD2'] = [0, 0]
        self.post_synaptic['VD3'] = [0, 0]
        self.post_synaptic['VD4'] = [0, 0]
        self.post_synaptic['VD5'] = [0, 0]
        self.post_synaptic['VD6'] = [0, 0]
        self.post_synaptic['VD7'] = [0, 0]
        self.post_synaptic['VD8'] = [0, 0]
        self.post_synaptic['VD9'] = [0, 0]


    # todo make class for brain with these as class variables
    def motorcontrol(self, ):
        #global body

        # accumulate left and right muscles and the accumulated values are
        # used to move the left and right motors of the robot

        for muscle in self.post_synaptic:  # if this doesn't work, do muscle in self.post_synaptic
            if muscle in self.mLeft:
                self.accumleft += self.post_synaptic[muscle][nextState]

                # print muscle, "Before", self.post_synaptic[muscle][thisState], accumleft
                # og version uses thisState to reset to 0
                # might be related to bug that would have infinite firing, in fireNeuron
                self.post_synaptic[muscle][nextState] = 0
                # print muscle, "After", self.post_synaptic[muscle][thisState], accumleft

            elif muscle in self.mRight:
                self.accumright += self.post_synaptic[muscle][nextState]

                # self.post_synaptic[muscle][thisState] = 0
                self.post_synaptic[muscle][nextState] = 0

        angle, mag = body.move(self.accumleft, self.accumright)
        accumleft = 0
        accumright = 0


    def dendrite_accumulate(self, dneuron):
        f = eval(dneuron)
        f()


    def fireNeuron(self, fneuron):
        # Could use DP for string -> function sigs rather than repeated Evals if this costs time
        # The threshold has been exceeded and we fire the neurite
        if fneuron != "MVULVA":
            f = eval(fneuron)
            f()
            # og version didn't have this
            # I think they added it b/c otherwise it would just have infinite firing
            # post_synaptic[fneuron][thisState] = 0
            self.post_synaptic[fneuron][nextState] = 0


    def runconnectome(self):
        """Each time a set of neuron is stimulated, this method will execute
            The weigted values are accumulated in the post_synaptic array
            Once the accumulation is read, we see what neurons are greater
            then the threshold and fire the neuron or muscle that has exceeded
            the threshold.
            """
        global thisState
        global nextState

        for ps in self.post_synaptic:
            if ps[:3] not in self.muscles and abs(self.post_synaptic[ps][thisState]) > self.threshold:
                self.fireNeuron(ps)
                # og version resets the entire postsynaptic array at this point
                # fucking why???
        self.motorcontrol()

        # swap from previous state to next state
        # this data structure could use some improvement
        for ps in self.post_synaptic:
            # if post_synaptic[ps][thisState] != 0:
            #         print ps
            #         print "Before Clone: ", post_synaptic[ps][thisState]

            # fired neurons keep getting reset to previous weight
            # wtf deepcopy -- So, the concern is that the deepcopy doesnt
            # scale up to larger neural networks??
            # I guess it wasn't working for them when they did it on the entire array?
            self.post_synaptic[ps][thisState] = copy.deepcopy(self.post_synaptic[ps][nextState])

            # this deep copy is not in the functioning version currently.
            # print "After Clone: ", post_synaptic[ps][thisState]

        thisState, nextState = nextState, thisState


    def trigger_food_sensors(self):
        self.dendrite_accumulate("ADFL")
        self.dendrite_accumulate("ADFR")

        self.dendrite_accumulate("ASGL")
        self.dendrite_accumulate("ASGR")
        self.dendrite_accumulate("ASIL")
        self.dendrite_accumulate("ASIR")

        self.dendrite_accumulate("ASJL")
        self.dendrite_accumulate("ASJR")

    def trigger_nose_touch_sensors(self):

        self.dendrite_accumulate("FLPR")
        self.dendrite_accumulate("FLPL")
        self.dendrite_accumulate("ASHL")
        self.dendrite_accumulate("ASHR")
        self.dendrite_accumulate("IL1VL")
        self.dendrite_accumulate("IL1VR")
        self.dendrite_accumulate("OLQDL")
        self.dendrite_accumulate("OLQDR")
        self.dendrite_accumulate("OLQVR")
        self.dendrite_accumulate("OLQVL")

    def trigger_anterior_harsh_touch_sensors(self):
        #untested, unused
        self.dendrite_accumulate("FLPL")
        self.dendrite_accumulate("FLPR")
        self.dendrite_accumulate("BDUL")
        self.dendrite_accumulate("BDUR")
        self.dendrite_accumulate("SDQR")

    def main(self):
        timestep_n = 5000000000000000000
        #timestep_n = 100000
        food_x = 200
        food_y = 0
        food_r = 50
        #while timestep < timestep_n if timestep_n > 0 else True:
        for timestep in tqdm(range(timestep_n)):
            #print(f"TIMESTEP: {timestep}")
            #if timestep % 6000 == 0:
            #body.clear()
            #turtle.update()


            # Check if it is bumping into the wall, and if so, trigger nose touch
            if body.nose_touching():
                body.cagecolor("black")
                body.pencolor("black")
                self.trigger_nose_touch_sensors()
                self.runconnectome()
            else:
                # Otherwise do nothing, unless we encounter food
                # todo we need to handle case where its on the wall and there's food
                #if timestep < 15 or body.distance(food_x,food_y) < food_r:
                if timestep < 15:
                    body.cagecolor("red")
                    body.pencolor("red")
                    self.trigger_food_sensors()
                    self.runconnectome()
                    if self.time_delays:
                        time.sleep(0.5)
                else:
                    body.cagecolor("blue")
                    body.pencolor("blue")
                    # no food sensors, but still run the brain
                    self.runconnectome()

        body.exit()

        print(np.mean(body.lefts))
        print(np.mean(body.rights))
        print(np.mean(body.rights)-np.mean(body.lefts))

def main():
    nematode = Nematode()
    nematode.main()

if __name__ == '__main__':
    main()


"""
OI FUTURE SELF

so we were super smart and stopped while we were refactoring to move the neurons to another system.

This is partly because we were thinking of changing to another brain altogether, but i digress.

let's finish refactoring and get the buddy running in the background, then once we get back in the groove of it
    we can decide again if we want to switch to another brain and start researching that, etc.
    
well, i've been improving it again

regardless of new brain or not, we were moving the neuron data structure to a weighted graph and we converged back
    into NEAT. So. Reminder I suppose that we need to convert this to some sort of data file most likely,
    rather than a hardcoded data structure with tons of neurons.
    
Oh, and we can make an endless mode if we feel like it still, where it just runs in the background.

But I see no reason why we couldn't just have NEAT be endless mode, plus that'd be much more interesting.


SO YEA PYTHON-NEAT WOO

"""