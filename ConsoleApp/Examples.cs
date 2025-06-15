using Core.Abstractions;
using Core.Mathematics;
using DataPipeline.Tokenization;

namespace ConsoleApp;
public static class Examples
{
    public static void AddSeperatorsToConsole()
    {
        Console.WriteLine(new string('=', 25));
        Console.WriteLine("Example Start");
        Console.WriteLine(new string('=', 25));
    }
    public static void RunModelConfigurationExample()
    {
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
    }
    public static void RunMathExample()
    {
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
    }
    public static void RunTokenizerExample()
    {
        // Create and test the tokenizer
        var tokenizer = new CharacterTokenizer();

        // Train on Shakespeare-style text
        const string trainingText = "To be or not to be, that is the question.";
        tokenizer.Fit(trainingText.AsSpan());

        Console.WriteLine($"Vocabulary size: {tokenizer.VocabularySize}");

        // Test encoding/decoding
        var tokens = tokenizer.Encode("Hello".AsSpan());
        var decoded = tokenizer.Decode(tokens.AsSpan());
        Console.WriteLine($"'Hello' -> {string.Join(",", tokens)} -> '{decoded}'");

        // Get vocabulary details
        var vocabInfo = tokenizer.GetVocabularyInfo();
        foreach (var charInfo in vocabInfo.Characters.Take(10))
        {
            Console.WriteLine($"Token {charInfo.TokenId}: {charInfo.DisplayName}");
        }
        Console.WriteLine($"Total unique characters: {vocabInfo.Characters.Count}");
        Console.WriteLine("End of tokenizer example");
    }
}
