#########################################
#########################################
############### Libraries ###############

from __future__ import print_function
import os
import neat
import csv
import random
import sys
import math
import numpy as np


#########################################
#########################################
### Fitness Threshold and Random Seed ###

TrainingThreshold = 100000.0
TestingThreshold = 1.0
SizeOfTrainingData = 0.7
Generations = 10000
counter = 0
sample = .25
TrainingGenerations = 50

#argument list
Arg_List = sys.argv
Directory = str(Arg_List[1])
Seed = int(Arg_List[2])
random.seed(Seed)

#########################################
#########################################
##### creating functions with a cap #####

def custom_square(x):
    #1.3*10^154 is the square root of the max float number
    if x > 1.3407807929942596e+154 or x < -1.3407807929942596e+154:
        x = 0
    return(x ** 2)

def custom_cube(x):
    #5.6*10^102 is the cube root of the max float number and 2.8*0^-103 is the cube root of the min float number
    if x > 5.643803094122288e+102 or x < 2.8126442852362986e-103:
        x = 0
    return(x ** 3)


#########################################
#########################################
#### Loading and splitting the data #####

#this opens the file with inputs
with open('Gold.csv') as f:
    data = [line for line in csv.reader(f)]
    header = data[0]
    content = [tuple(map(float, line)) for line in data[1:]]

#this section creates the input list
Input_List = []
for input in range(len(content)-1):
    Input_List.append(tuple(content[input]))

#this opens the file with the expected outputs
with open('GoldExpectedOutputs.csv') as a:
    data2 = [line for line in csv.reader(a)]
    header2 = data2[0]
    output_content = [tuple(map(float,line)) for line in data2[1:]]

#this section creates the output list
Output_List = []

CSV_Output_List = []
for out in range(1,len(content)):
    Output = [output_content[out][0]]
    Output_List.append(tuple(Output))
    CSV_Output_List.append(Output[0])
print(len(Output_List))
#this splits the inputs
split = int(SizeOfTrainingData * len(Input_List))
Training_Input = Input_List[:split]
Testing_Input = Input_List[split:]
TrainingSetSize = len(Training_Input)


#this splits the outputs
Training_Output = Output_List[:split]
Testing_Output= Output_List[split:]
CSV_Output = CSV_Output_List[split:]

print(Training_Input[0])
print(Training_Output[0])