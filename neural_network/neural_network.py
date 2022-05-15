
import numpy as np
import random
import functions
import activations
from matplotlib import pyplot as plt

class NeuralNetwork(object):

    data = list()
    # a list of tuples containing the inputs and the outputs
    ACTIVATIONS = {
        'relu': [activations.relu,activations.relu_derivative],
        'sigmoid':[activations.sigmoid,activations.sigmoid_derivative],
        'liniar':[activations.liniar,activations.liniar_derivative],
        'leaky':[activations.leaky_relu, activations.leaky_relu_derivative]
    }

    function = list()
    derivative = list()
    # hidden activation functions and derivatives for each layer of the network


    neurons = None
    # number of neurons on each layers
    weights = list()
    biases = list()
    gradients = list()
    learning_rate = 0.0001

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
        self.initialize_weights()
        self.initialize_biases()
    

    def initialize_weights(self):
        self.weights.append(np.random.randn(1,self.neurons[0])/15)
        #input to first hidden layer
        for i in range(1,self.layers):
            self.weights.append(np.random.randn(self.neurons[i-1],self.neurons[i])/15)
        #between hidden layers
        self.weights.append(np.random.randn(self.neurons[self.layers-1],1)/5)
        

    def initialize_biases(self):
        for i in self.neurons:
            self.biases.append(np.random.randn(i))
        

    def set_funtions(self, *args) -> None:
        if len(args) != self.layers + 1 :
            raise("Wrong number of function parameters")
        for i in args:
            if not i in self.ACTIVATIONS:
                raise("Wrong function name")
            self.function.append(self.ACTIVATIONS[i][0])
            self.derivative.append(self.ACTIVATIONS[i][1])
        

            

    def add_data():
        pass


    def generate_data(self, left:int, right:int, fnc:function):
        
        for i in range(100000):
            x = random.uniform(left,right)   
            fx = fnc(x)
            self.data.append([x,fx])   
        self.data = np.array(self.data)
        self.data = np.split(self.data,len(self.data)/50)
        # generates batches of 10 inputs/outputs pairs


    def forward_propagation(self, input) -> float:
        
        output = list()
        output.append(np.array([[input]]))
        
        #print(output[0])

        for i in range(1,self.layers+1): 
            output_i = np.array(output[i-1]@self.weights[i-1])
            output_i = output_i + self.biases[i-1]
            # calculates dot product between weights and inputs
            output_i = [self.function[i-1](j) for j in output_i[0]]
            output.append([output_i])


        output_i = output[self.layers]@self.weights[self.layers]
        #output layer has liniar activation in this case
        output.append(output_i)
        #output layer
        self.output = output

        #print(output[3])

        return output[self.layers+1]

        

    def backpropagation(self):
        #computing gradient for each weight with respect to the cost function
        #gradient_wk = self.output[]



        for i in range (self.layers,0,-1):
            pass


    def calculate_gradients(self, output_derivative):
        output = self.output[self.layers+1][0][0]
        pred_grad = np.array([[self.derivative[self.layers](output)]])
        # derivative of weighted sums ZL with respect to y_hat (predicted output)
        pred_grad = pred_grad * output_derivative
        # multiply with derivative of cost with respect to y_hat
        previous_gradient = pred_grad

        gradient_output_weights = (np.array(self.output[self.layers]).T)@(previous_gradient.T)
        self.gradients = list()
        self.gradients.append(gradient_output_weights)
        
        for i in range(self.layers,0,-1):
            #print(previous_gradient)
            new_gradient = self.weights[i]@previous_gradient 
            #derivative of Cost with respect with A l-1 -> needed to backprop further
            
            
            output_d = np.array([[self.derivative[i-1](j)] for j in self.output[i][0]])
            previous_gradient = new_gradient * output_d
            #computing @C/@aK L-1 * @aK L-1/@zK L-1
            #print(previous_gradient)
            #print(self.output[i-1])
            gradient_output_weights = (np.array(self.output[i-1]).T)@(previous_gradient.T)
            self.gradients.append(gradient_output_weights)
            #print(gradient_output_weights)

    
        fixed_gradient = list()
        #inversting gradients to have the same shape as the weights
        for i in range(self.layers,-1,-1):
            fixed_gradient.append(self.gradients[i])
        
        self.gradients = fixed_gradient
        #print(self.weights)
        for i in range(0,self.layers+1):
            
            #print(i, end="inceput -+++++++++++ \n")
            self.gradients[i] = self.gradients[i]*self.learning_rate
            #print(self.gradients[i])
            #print("stop ************")
            #print(self.weights[i])
            self.weights[i] -= self.gradients[i]
            #print(i, end="")
            #print("---------------")




        
    def run(self):
        i = 0
        for epoch in self.data:
            i+=1
            cost = 0
            output_grad = 0 # means of output gradients with respect to 
            for element in epoch:
                y_hat = self.forward_propagation(element[0]) # forward prop given the input
                y = element[1]
                cost += 1/2 * (y_hat - y)**2
                output_derivative = (y_hat - y) # output derivative with respect to y_hat
                self.calculate_gradients(output_derivative)
                #print(cost)

                # 1/2 for derivative purposes
            cost = cost/(len(epoch))
            #plt.plot(epoch,cost[0][0],marker='o')
            #print("batch cost")
            #print(cost[0][0])
            plt.plot(i,cost[0][0],marker='o')
            print(cost[0][0])
        plt.show()


def main():
    nn = NeuralNetwork(4,15,10,5,2)
    nn.generate_data(-5,5,functions.func5)
    nn.set_funtions('leaky','leaky','leaky','leaky','liniar') 
    # last activation is liniar since approximating a function is a regression problem
    nn.run() 
  
    
    pass



if __name__ == "__main__":
    main()