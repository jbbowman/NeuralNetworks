import numpy as np

class LogisticRegression:
    def __init__(self, weights, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = weights
        self.bias = 0.5

    def calc_weighted_sum(self, feature):
        return np.dot(feature, self.weights) + self.bias

    @staticmethod
    def calc_sigmoid(weighted_sum):
        return 1 / (1 + np.exp(-weighted_sum))

    @staticmethod
    def calc_cross_entropy(target, prediction):
        return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))

    def update_weights(self, target, prediction, feature):
        new_weights = []
        for value, weight in zip(feature, self.weights):
            new_weight = weight + self.learning_rate * (target - prediction) * value
            new_weights.append(new_weight)
        return new_weights

    def update_bias(self, target, prediction):
        return self.bias + self.learning_rate * (target - prediction)

    def initialize_neural_net(self, features, targets):
        individual_loss = []
        for feature, target in zip(features, targets):
            weighted_sum = self.calc_weighted_sum(feature)
            prediction = self.calc_sigmoid(weighted_sum)
            loss = self.calc_cross_entropy(target, prediction)
            individual_loss.append(loss)
        return individual_loss

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

if __name__ == '__main__':
    from customers import *

    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    features, targets, weights = prep_data(data)
    model = LogisticRegression(weights, 0.1, 30)
    individual_loss = model.initialize_neural_net(features, targets)

    print(individual_loss)
  
