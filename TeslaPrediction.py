#########################################
#########################################
############### Libraries ###############

from __future__ import print_function
import os
import neat
import csv
import random
import sys

#########################################
#########################################
### Fitness Threshold and Random Seed ###

TrainingThreshold = 10000
TestingThreshold = 5000
SizeOfTrainingData = 0.7
Generations = 15000
counter = 0


#argument list
Arg_List = sys.argv
Directory = str(Arg_List[1])
Seed = int(Arg_List[2])
random.seed(Seed)




#########################################
#########################################
#### Loading and splitting the data #####

#this opens the file with inputs
with open('OneInputExperiment.csv') as f:
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

#this splits the outputs
split2 = int(SizeOfTrainingData * len(Output_List))
Training_Output = Output_List[:split2]
Testing_Output= Output_List[split2:]
CSV_Output = CSV_Output_List[split2:]


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

#########################################
#########################################
##Fitness Function for Training Dataset##

def eval_genomes(genomes, config):
    
    #This for loop will run every genome and determine its fitness
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(Training_Input, Training_Output):
            output = net.activate(xi)
            genome.fitness -= abs(output[0] - xo[0])

#########################################
#########################################
####### creating the run function #######

def run(config_file):
    global counter
    # These lines load in the configuration file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Initializing the training and testing error variables
    training_error = 100
    testing_error = 100000

    while training_error >= TrainingThreshold or abs(testing_error) >= TestingThreshold:

        #These lines will shuffle the training data set every n generations
        random.shuffle(Training_Data)
        Training_Input, Training_Output = zip(*Training_Data)
        # This line is initializing the testing error at zero
        testing_error = 0
        # Run for up to 1000 generations.
        winner = p.run(eval_genomes, 100)
        # This line is setting the training error equal to the error of the best genome
        training_error = abs(winner.fitness)

        #########################################
        #########################################
        ##Fitness Function For Testing Dataset ##  

        # This if statement checks to see if the training error is low enough and will 
        # run the best genome on the testing data
        if training_error <= TrainingThreshold:
            testing_error = 0
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            Outputs = []
            # This for loop runs the best genome on the testing data
            for ti, to in zip(Testing_Input, Testing_Output): 
                Test_Output = winner_net.activate(ti)
                testing_error -= abs(Test_Output[0] - to[0])
                Outputs.append(Test_Output[0])

            # This line will add a new row to the CSV containing all of the outputs from the testing data
            writer2.writerow(Outputs)    
        # This line will add a new row to the CSV containing the average errors for the training and testing datasets 
        writer1.writerow([training_error/len(Training_Input), abs(testing_error)/len(Testing_Input)])
        A.flush()
        B.flush()
        

        
        counter += 100
        print('Generation=', counter)
        if counter > Generations:
            break
           


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