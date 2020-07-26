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
with open('cleantsladata.csv') as f:
    data = [line for line in csv.reader(f)]
    header = data[0]
    content = [tuple(map(float, line)) for line in data[1:]]

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




# This line prepares the training data to be shuffled
Training_Data = list(zip(Training_Input,Training_Output))




#########################################
#########################################
################## CSV ##################

# This CSV keeps track of the average training and testing error
A = open(Directory+'AverageError.csv','w')
writer1 = csv.writer(A)
ErrorColumns = ['AVG Training Error']+['AVG Testing Error']
writer1.writerow(ErrorColumns)
A.flush()

# This CSV will keep track of the expected and actual outputs
B = open(Directory+'TestingOutputs.csv','w')
writer2 = csv.writer(B)
# The first line of the CSV will contain the expected outputs
writer2.writerow(CSV_Output)
B.flush()

#this line creates the text file containing the best genome structure
GenomeStructure = open(Directory+'GenomeStructure.txt','w')

#this line will create a csv that will track population statistics
C = open(Directory+'PopulationStats.csv','w')
writer3 = csv.writer(C)
Columns = ['NumberOfSpecies']
writer3.writerow(Columns)
C.flush()

#########################################
#########################################
#####Species data collection function####

def data_collection(stats,gens):
    species_sizes = stats.get_species_sizes()
    randomlist = range(1,gens)
    for generations in randomlist:
        #for ever generation count the number of unique populations in a list and write them
        writer3.writerow([len(np.unique(species_sizes[generations]))])
        C.flush()








#########################################
#########################################
##Fitness Function for Training Dataset##

def eval_genomes(genomes, config):
    

    #This for loop will run every genome and determine its fitness
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(Training_Input, Training_Output):
            output = net.activate(xi)
            #genome.fitness -= abs(xo[0] - output[0])
            #1 stock price is going up #0 stock price is going down
            if output[0] >= 0.5:
                output[0] = 1.0
            else:
                output[0] = 0.0
            #punishing type 1 error
            if xo[0] == 0 and output[0] == 1:
                genome.fitness -= 5.0
            #punishing type 2 error
            if xo[0] == 1 and output[0] == 0:
                genome.fitness -= 3.0
            

           
#########################################
#########################################
####### creating the run function #######

def run(config_file):
    global counter
    
    # These lines load in the configuration file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    


    # These two lines add the custom functions with capped values to the config file
    # These lines must come before the creation of the population
    config.genome_config.add_activation('my_square_function', custom_square)
    config.genome_config.add_activation('my_cube_function', custom_cube)
    
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)


    #stats functions
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Initializing the training and testing error variables
    training_error = 100
    testing_error = 100000

    while training_error >= TrainingThreshold or abs(testing_error) >= TestingThreshold:

        #These lines will shuffle the training data set every n generations
        random.shuffle(Training_Data)
        Training_Input, Training_Output = zip(*Training_Data)
        
        # This line is initializing the testing error at zero
        testing_error = 0

        
        #These three lines sample the first n% of the shuffled data
        split2 = int(sample * len(Training_Input))
        Training_Input = Training_Input[:split2]
        Training_Output = Training_Output[:split2]
       

        # Run for up to 100 generations.
        winner = p.run(eval_genomes, TrainingGenerations)
        



        # This line is setting the training error equal to the error of the best genome
        training_error = abs(winner.fitness)

        #########################################
        #########################################
        ##Fitness Function For Testing Dataset ##  

        # This if statement checks to see if the training error is low enough and will 
        # run the best genome on the testing data
        if training_error <= TrainingThreshold:
            testing_error = 0
            winner_net = neat.nn.RecurrentNetwork.create(winner, config)
            Outputs = []
            # This for loop runs the best genome on the testing data

            for ti, to in zip(Testing_Input, Testing_Output): 
                Test_Output = winner_net.activate(ti)
                if Test_Output[0] >= 0.5:
                    Test_Output[0] = 1.0
                else:
                    Test_Output[0] = 0.0
                #punishing type 1 error
                if to[0] == 0 and Test_Output[0] == 1:
                    testing_error -= 5.0
                #punishing type 2 error
                if to[0] == 1 and Test_Output[0] == 0:
                    testing_error -= 3.0

                #testing_error -= abs(to[0] - Test_Output[0])
                Outputs.append(Test_Output[0])

            # This line will add a new row to the CSV containing all of the outputs from the testing data
            writer2.writerow(Outputs)


        # This line will add a new row to the CSV containing the average errors for the training and testing datasets 
        writer1.writerow([training_error/TrainingSetSize, abs(testing_error)/len(Testing_Input)])
        A.flush()
        B.flush()
        
        counter += 50
        print('Generation=', counter)
        if counter == Generations:

            #this line keeps track of the best genome structure
            GenomeStructure.write('\nBest genome:\n{!s}'.format(winner))
            
            #this line will stop the code
            break
    #function that collects species data
    data_collection(stats,Generations)
            
#############################################
#############################################
###### Determining path to config file ######

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_file.txt')
    run(config_path)

#############################################
#############################################
#############################################

