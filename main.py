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
