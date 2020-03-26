#########################################
#########################################
############### Libraries ###############

from __future__ import print_function
import os
import neat
import csv
import random

#########################################
#########################################
### Fitness Threshold and Random Seed ###

FitnessThreshold = 1000000000.0
random.seed(1)

#########################################
#########################################
#### Loading and splitting the data #####

with open('UpOrDown.csv') as f:
    data = [line for line in csv.reader(f)]
    header = data[0]
    content = [tuple(map(float, line)) for line in data[1:]]

#this section creates the input list
Input_List = []
for i in range(len(content)-2):
    Value1 = list(content[i])
    Value2 = list(content[i+1])
    New_Tuple = tuple(Value1 + Value2)
    Input_List.append(New_Tuple)

#this section creates the output list
Output_List = []
for out in range(2,len(content)):
    Output = [content[out][0]]
    Output_List.append(tuple(Output))
    
#this splits the inputs
split = int(0.7 * len(Input_List))
training_Input = Input_List[:split]
testing_Input = Input_List[split:]

#this splits the outputs
split2 = int(0.7 * len(Output_List))
training_Output = Output_List[:split2]
testing_Output= Output_List[split2:]

#########################################
#########################################
##Fitness Function for Training Dataset##

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for xi, xo in zip(training_Input, training_Output):
            guess = 0
            output = net.activate(xi)
            if output[0] > 0:
                guess = 1
            fitness -= abs(guess - xo[0])
        genome.fitness -= fitness ** 2
            
#########################################
#########################################
####### creating the run function #######

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    training_error = 100
    testing_error = 100
    
    while training_error >= FitnessThreshold or testing_error >= 10.0:
        testing_error = 100

        # Run for up to 300 generations.
        winner = p.run(eval_genomes, 100)
        
        # print the key for the winner
        print('WINNER=',winner)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
    
        training_error = abs(winner.fitness)

#########################################
#########################################
##Fitness Function For Testing Dataset ##    #look into maybe turning this into its own function

        if training_error < FitnessThreshold:
            testing_error = 0
            test = 0
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            for ti, to in zip(testing_Input, testing_Output): 
                test_guess = 0
                testing_output = winner_net.activate(ti)
                if testing_output[0] > 0:
                    test_guess = 1
                test += test_guess - to[0]
                print("expected output {!r}, got {!r}".format(to, test_guess))
            testing_error -= test ** 2 
            print('TRAINING ERROR=',training_error)
            print('TESTING ERROR=',abs(testing_error))
    
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