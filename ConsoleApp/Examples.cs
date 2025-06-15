using Core.Abstractions;
using Core.Mathematics;

namespace ConsoleApp;
public static class Examples
{
    public static void RunModelConfigurationExample()
    {
        Console.WriteLine("Start of model configuration example");
        var config = new ModelConfiguration
        {
            EmbeddingDim = 256,
            VocabularySize = 100,
            ContextLength = 128,
            NumLayers = 4,
            NumAttentionHeads = 8,
            FeedForwardDim = 1024
        };
        config.Validate(); // Should not throw
        var paramCount = config.CalculateParameterCount();
        Console.WriteLine($"Model will have {paramCount:N0} parameters");
        Console.WriteLine("End of model configuration example");
        Console.ReadKey();
    }
    public static void RunMathExample()
    {
        Console.WriteLine("Start of math example");
        // Test basic operations
        var a = new float[] { 1, 2, 3, 4 }; // 2x2 matrix
        var b = new float[] { 5, 6, 7, 8 }; // 2x2 matrix  
        var result = new float[4];

        MatrixOperations.MatrixMultiply(a, b, result, 2, 2, 2);
        Console.WriteLine($"Matrix multiply result: [{string.Join(", ", result)}]");

        // Test softmax
        var logits = new float[] { 2.0f, 1.0f, 0.1f };
        var probs = new float[3];
        NumericalFunctions.Softmax(logits, probs);
        Console.WriteLine($"Softmax: [{string.Join(", ", probs.Select(p => p.ToString("F3")))}]");
        Console.WriteLine($"Sum: {probs.Sum():F6}"); // Should be 1.0
        Console.WriteLine("End of math example");
        Console.ReadKey();
    }
}
