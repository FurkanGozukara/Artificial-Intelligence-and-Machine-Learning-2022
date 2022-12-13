// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;
using System;
internal class Program
{
    static void Main(string[] args)
    {
        // Create a list of input sentences
        List<string> sentences = new List<string>
            {
                "This is the first sentence.",
                "This is the second sentence.",
                "This sentence is similar to the first sentence.",
                "Is this the first sentence?",
                "The dog chased the cat."
            };

        // Create a dictionary to store the TD-IDF values for each sentence
        Dictionary<string, double> tdIdfValues = new Dictionary<string, double>();

        // Calculate the TD-IDF values for each sentence
        foreach (string sentence in sentences)
        {
            // Split the sentence into words (tokenize it)
            string[] words = sentence.Split(new char[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            // Calculate the TD-IDF value for this sentence
            double tdIdfValue = CalculateTdIdf(sentence, sentences, words);

            // Add the TD-IDF value for this sentence to the dictionary
            tdIdfValues[sentence] = tdIdfValue;
        }

        // Print the TD-IDF values for each sentence
        foreach (KeyValuePair<string, double> kvp in tdIdfValues)
        {
            Console.WriteLine("Sentence: {0} TD-IDF: {1}", kvp.Key, kvp.Value);
        }
    }

    static double CalculateTdIdf(string sentence2, List<string> sentences, string[] words)
    {
        double td = 0.0; // Term frequency
        double idf = 0.0; // Inverse document frequency

        // Calculate the term frequency (TF)
        foreach (string word in words)
        {
            // Count the number of times this word appears in the sentence
            int count = 0;
            foreach (string w in words)
            {
                if (w == word)
                {
                    count++;
                }
            }

            // Calculate the TD value for this word
            double tdValue = count / (double)words.Length;

            // Add the TD value to the total TD value for the sentence
            td += tdValue;
        }

        // Calculate the inverse document frequency (IDF)
        foreach (string word in words)
        {
            // Count the number of sentences that contain this word
            int count = 0;
            foreach (string sentence in sentences)
            {
                string[] sentenceWords = sentence.Split(new char[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (string w in sentenceWords)
                {
                    if (w == word)
                    {
                        count++;
                        break;
                    }
                }
            }

            // Calculate the IDF value for this word
            double idfValue = Math.Log10(sentences.Count / (double)count);

            idf += idfValue;
        }
        double tdIdf = td * idf;

        // Return the TD-IDF value
        return tdIdf;
    }
}