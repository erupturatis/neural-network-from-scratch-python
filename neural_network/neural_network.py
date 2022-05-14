import numpy as np
import random
import functions
import activations

class NeuralNetwork(object):

    data = list()
    # a list of tuples containing the inputs and the outputs

    function = list()
    derivative = list()
    # hidden activation functions and derivatives for each layer of the network


    neurons = None
    # number of neurons on each layers

    def print_stuff(self, *args,):
        print(*args)

    def write(text:str,filename:str = "file.txt"):
        file = open("text.txt",'w')
        file.write(text)
        file.close()

    # auxiliary functions for output



    def __init__(self, layers, *args) -> None:
        self.layers = layers
        self.neurons = [*args]
    

    def set_funtions(self, *args) -> None:
        if len(args) > self.layers :
            raise("Too many function function parameters")
            

    def add_data():
        pass


    def generate_data(self, left:int, right:int, fnc:function):
        
        for i in range(1000):
            x = random.uniform(left,right)   
            fx = fnc(x)
            self.data.append((x,fx))   
        # print(self.data)


    def forward_propagation(self):
        pass


    def create_gradients(self):
        pass


    def backpropagation(self):
        pass




def main():
    nn = NeuralNetwork(2,5,5)
    nn.generate_data(-6,2,functions.func3)
    pass



if __name__ == "__main__":
    main()