﻿
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace MLModel1_ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            MLModel1.ModelInput sampleData = new MLModel1.ModelInput()
            {
                Col0 = 5.8F,
                Col1 = 2.7F,
                Col2 = 4.1F,
                Col3 = 1F,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = MLModel1.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Col4 with predicted Col4 from sample data...\n\n");


            Console.WriteLine($"Col0: {5.8F}");
            Console.WriteLine($"Col1: {2.7F}");
            Console.WriteLine($"Col2: {4.1F}");
            Console.WriteLine($"Col3: {1F}");
            Console.WriteLine($"Col4: {1F}");


            Console.WriteLine($"\n\nPredicted Col4: {predictionResult.Prediction}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
