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




#########################################
#########################################
#### Loading and splitting the data #####

#this opens the file with inputs
with open('UnNormalized.csv') as f:
    data = [line for line in csv.reader(f)]
    header = data[0]
    content = [tuple(map(float, line)) for line in data[2:]]
     
#this section creates the input list
Input_List = []
for input in range(len(content)-1):
    Input_List.append(tuple(content[input]))

#this opens the file with the expected outputs
with open('tslaexpectedoutputs.csv') as a:
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