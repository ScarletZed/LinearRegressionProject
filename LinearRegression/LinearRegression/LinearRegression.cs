using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinearRegression
{
    public class LinearRegression
    {
        private double[] coefficients; // Coefficients for the linear model (slope, intercept)

        public void Train(double[] X, double[] y, double learningRate, int epochs)
        {
            if (X.Length != y.Length)
                throw new ArgumentException("Input array lengths must match.");

            int numSamples = X.Length;
            int numFeatures = 1; // For simplicity, we'll consider only one feature (univariate)

            // Initialize coefficients
            coefficients = new double[numFeatures + 1]; // One for the slope, one for the intercept

            // Add a column of 1s to X for the intercept term
            double[][] XWithIntercept = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                XWithIntercept[i] = new double[] { 1.0, X[i] };
            }

            // Gradient Descent
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] predictions = new double[numSamples];

                // Compute predictions for each sample
                for (int i = 0; i < numSamples; i++)
                {
                    double prediction = 0;
                    for (int j = 0; j < numFeatures + 1; j++)
                    {
                        prediction += coefficients[j] * XWithIntercept[i][j];
                    }
                    predictions[i] = prediction;
                }

                // Compute error and update coefficients
                for (int j = 0; j < numFeatures + 1; j++)
                {
                    double gradient = 0;
                    for (int i = 0; i < numSamples; i++)
                    {
                        gradient += (predictions[i] - y[i]) * XWithIntercept[i][j];
                    }
                    coefficients[j] -= learningRate * (gradient / numSamples);
                }
            }
        }

        public double Predict(double x)
        {
            if (coefficients == null)
                throw new InvalidOperationException("The model has not been trained.");

            return coefficients[0] + coefficients[1] * x;
        }
    }
}
