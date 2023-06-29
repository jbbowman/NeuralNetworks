# NeuralNetworksIntro

## Background
The following repository describes how to build and train a machine learning algorithm. I will do this with a logistic regression model, which can be seen as a simple, single-layer neural network with one neuron (or perceptron).

## Introduction
Suppose a bank wants to predict whether a customer is likely to default on their loan based on their age, credit score, debt-to-income ratio, and loan-to-asset value ratio. The bank has this information for previous borrowers as well as the outcome of their loan. Using this dataset, the bank can use a logistic regression model to predict whether a new customer is likely to default on a loan or not. Before developing our logistic regression model, the code in customers.py generates customers, a data framework, and data in a prepared format to test and develop our model.

### customers.py Output
Code:
```Python
if __name__ == '__main__':
    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    prepped_data = prep_data(data)

    print(customer_data, data, prepped_data, sep="\n\n")
```
Output:
```Output
[[0.6436363636363637, 0.633090468982135, 0.10385785585637752, 0.9333333333333333, 0], [0.23454545454545456, 0.6169783228052368, 0.11853714815446115, 0.30666666666666664, 0], [0.11454545454545455, 0.46611204163509357, 0.9958532078406778, 0.6533333333333333, 1], [0.10909090909090909, 0.6516349798529134, 0.49722891199630226, 0.92, 1], [0.8127272727272727, 0.9122495953119779, 0.7318774525822924, 0.8933333333333333, 1]]

   Credit Score  Debt/Income  Loan/Value       Age  Default
0      0.643636     0.633090    0.103858  0.933333        0
1      0.234545     0.616978    0.118537  0.306667        0
2      0.114545     0.466112    0.995853  0.653333        1
3      0.109091     0.651635    0.497229  0.920000        1
4      0.812727     0.912250    0.731877  0.893333        1

(array([[0.64363636, 0.63309047, 0.10385786, 0.93333333],
       [0.23454545, 0.61697832, 0.11853715, 0.30666667],
       [0.11454545, 0.46611204, 0.99585321, 0.65333333],
       [0.10909091, 0.65163498, 0.49722891, 0.92      ],
       [0.81272727, 0.9122496 , 0.73187745, 0.89333333]]), array([0, 0, 1, 1, 1], dtype=int64), array([0.00624203, 0.39826641, 0.94854967, 0.37765307]))
```

## Building a Neural Network in Python
After generating customer data, we can begin developing our logistic regression model. This involves developing our perceptron and integrating an error function, which in this case will use cross entropy loss; it will process the customer features (age, credit score, etc.) and output a binary value, indicating whether a customer is likely to default on their loan. logistic_regression.py displays a functioning perceptron and error function.

### logistic_regression.py Description
constructor: Initializes the model with the provided weights, learning_rate, and epochs (number of times the model will go through the entire dataset). The weights represent the importance of each input feature, and the bias is an additional term that allows the model to better fit the data by shifting it.
calc_weighted_sum: Takes a single customer's feature data and calculates the weighted sum by multiplying each feature with its corresponding weight and adding the bias. 
calc_sigmoid: Takes the weighted sum as input and calculates the sigmoid function value, which will give a probability value between 0 and 1. This is also known as the perceptron's activation function.
calc_cross_entropy: Calculates the cross-entropy loss between the model's prediction and the actual outcome. Cross-entropy loss is an error function that quantifies the difference between the predicted probabilities and the actual outcomes.
update_weights: Updates the weights of the model based on the difference between the actual outcome and the prediction. This method uses the learning rate and the feature data to make the updates. The learning rate is a hyperparameter that controls how much the weights are adjusted during each update. A smaller learning rate ensures that the model converges to a solution more slowly but with more precision, while a larger learning rate may cause the model to converge faster but with less precision.
update_bias: Updates the bias term of the model, using the same learning rate and the difference between the actual outcome and the prediction.
initialize_neural_net: Takes the features and targets as input and initializes the network It calculates the weighted sum, the prediction, and the cross-entropy loss for each customer. It then returns a list of individual losses for each customer, which can be used to assess the model's performance and update the weights and bias accordingly.

### logistic_regression.py Output
Code:
```Python
if __name__ == '__main__':
    from customers import *

    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    features, targets, weights = prep_data(data)
    model = LogisticRegression(weights, 0.1, 30)
    individual_loss = model.initialize_neural_net(features, targets)

    print(individual_loss)
```
Output:
```Output
[0.7230167195909061, 0.08413734833010549, 0.03392163449503953, 0.09488236539713085, 0.04974681478844909]
```

## Training a Neural Network in Python
After developing the perceptron and error function, we can begin training our neural network, in our case, using gradient descent, to minimize the value of the error function. Here is the code for the training part of the model:
```Python
def train_neural_net(self, features, targets):  # uses gradient descent
    epoch_loss = []
    for e in range(self.epochs):
        individual_loss = self.initialize_neural_net(features, targets)

        for feature, target, loss in zip(features, targets, individual_loss):
            weighted_sum = self.calc_weighted_sum(feature)
            prediction = self.calc_sigmoid(weighted_sum)

            self.weights = self.update_weights(target, prediction, feature)
            self.bias = self.update_bias(target, prediction)

        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
    return epoch_loss
```

Using our perceptron and error function once produces one epoch. We are looking to produce multiple epochs that converge toward a value of zero in our error function. We do this by calling our initialize_neural_net function for each epoch; we this in the first for loop. The next for loop iterates through the features, targets, and individual_loss lists simultaneously using the zip function. For each customer, the following steps are performed:
1.	The weighted sum is calculated using the calc_weighted_sum method.
2.	The prediction is calculated using the calc_sigmoid method.
3.	The weights are updated using the update_weights method, which uses the difference between the target and prediction, the learning rate, and the input features.
4.	The bias is updated using the update_bias method, which uses the difference between the target and prediction and the learning rate.
After iterating through all customers, the average loss for the current epoch is calculated by summing up the individual losses and dividing by the total number of customers. This average loss is then appended to the epoch_loss list. Once all epochs are completed, the epoch_loss list is returned. This list contains the average cross-entropy loss for each epoch, which can be used to evaluate the model's performance over time.
Here is the main function that calls all the necessary methods to create a holistic model:
main.py

```Python
from customers import *
from logistic_regression import *

def main():
    # create bogus data
    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    features, targets, weights = prep_data(data)

    # create logistic regression model
    model = LogisticRegression(weights, 0.1, 30)
    epoch_loss = model.train_neural_net(features, targets)

    # print results
    for e in range(len(epoch_loss)):
        print("**************************")
        print("Epoch:", e)
        print("Average Loss:", epoch_loss[e])

if __name__ == '__main__':
    main()
```

```Output
Sample runs:
**************************
Epoch: 0
Average Loss: 0.43088403301379835
**************************
Epoch: 1
Average Loss: 0.3708959038317227
**************************
Epoch: 2
Average Loss: 0.33133881819395056
**************************
Epoch: 3
Average Loss: 0.30656790380136645
**************************
Epoch: 4
Average Loss: 0.2913697411967414
**************************
Epoch: 5
Average Loss: 0.2819538440384622
**************************
Epoch: 6
Average Loss: 0.2759002513153225
**************************
Epoch: 7
Average Loss: 0.27176615506077245
**************************
Epoch: 8
Average Loss: 0.2687201396846688
**************************
Epoch: 9
Average Loss: 0.26629073024519345
**************************
Epoch: 10
Average Loss: 0.2642125583178507
**************************
Epoch: 11
Average Loss: 0.26233689161736845
**************************
Epoch: 12
Average Loss: 0.2605806994423558
**************************
Epoch: 13
Average Loss: 0.2588978843966283
**************************
Epoch: 14
Average Loss: 0.25726306608594507
**************************
Epoch: 15
Average Loss: 0.25566243995517524
**************************
Epoch: 16
Average Loss: 0.2540886247928309
**************************
Epoch: 17
Average Loss: 0.25253776271005346
**************************
Epoch: 18
Average Loss: 0.25100789308321614
**************************
Epoch: 19
Average Loss: 0.2494980473021661
**************************
Epoch: 20
Average Loss: 0.2480077507574082
**************************
Epoch: 21
Average Loss: 0.24653675400966465
**************************
Epoch: 22
Average Loss: 0.24508489201283162
**************************
Epoch: 23
Average Loss: 0.24365201405127362
**************************
Epoch: 24
Average Loss: 0.24223795201401108
**************************
Epoch: 25
Average Loss: 0.2408425088546986
**************************
Epoch: 26
Average Loss: 0.2394654571778238
**************************
Epoch: 27
Average Loss: 0.23810654247408025
**************************
Epoch: 28
Average Loss: 0.236765488104486
**************************
Epoch: 29
Average Loss: 0.23544200056546397
```

Our model begins to flatten at around .23. A lower value indicates that weights and bias have been properly adjusted to produce a more accurate model. The closer it is to zero, the more confident one can be of its accuracy.

## Conclusion
This paper has demonstrated how to implement a logistic regression model for binary classification using a simple, single-layer neural network with one neuron (perceptron). The program combines customer data generation with model training and evaluation, providing a complete implementation of the logistic regression algorithm.
