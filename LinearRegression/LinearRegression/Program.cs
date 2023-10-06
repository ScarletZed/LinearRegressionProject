using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinearRegression
{
    internal class Program
    {
        static void Main(string[] args)
        {
            double[] X = { 1, 2, 3, 4, 5 };
            double[] y = { 3, 6, 8, 11, 14 };

            LinearRegression model = new LinearRegression();
            model.Train(X, y, learningRate: 0.01, epochs: 1000);

            double prediction = model.Predict(6);
            Console.WriteLine("Prediction for x = 6: " + prediction);
        }
    }
}
