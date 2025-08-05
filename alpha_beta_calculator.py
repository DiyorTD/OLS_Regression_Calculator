import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_ols(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    beta_hat = numerator / denominator  # Covariance divided by Variance
    alpha_hat = y_mean - beta_hat * x_mean
    return alpha_hat, beta_hat

def predict(X, alpha_hat, beta_hat):
    return alpha_hat + beta_hat * X

def main():
    print("Enter data separated by commas")
    X_input = input("Enter X values (independent variable): ")
    Y_input = input("Enter Y values (dependent variable): ")

    # Convert input strings to numeric NumPy arrays
    X = np.array([float(x.strip()) for x in X_input.split(',')])
    Y = np.array([float(y.strip()) for y in Y_input.split(',')])

    # OLS Estimation
    alpha_hat, beta_hat = calculate_ols(X, Y)
    print(f"\nOLS Estimates:\nIntercept (alpha) = {alpha_hat:.4f}\nSlope (beta) = {beta_hat:.4f}")

    # Predictions
    Y_pred = predict(X, alpha_hat, beta_hat)

    # Plot Actual Data and Fitted Regression Line
    plt.scatter(X, Y, color='red', label="Actual Data")
    plt.plot(X, Y_pred, color='blue', label="Fitted Line")
    plt.xlabel("X Variable")
    plt.ylabel("Y Variable")
    plt.title("OLS Regression Line")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print Residuals
    residuals = Y - Y_pred
    print("\nResiduals:")
    for i, res in enumerate(residuals):
        print(f"Data Point {i+1}: Residual = {res:.4f}")

if __name__ == "__main__":
    main()
