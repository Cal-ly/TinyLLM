using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infrastructure.Tokenization;
/// <summary>
/// Extension methods for working with tokenizers
/// </summary>
public static class TokenizationExtensions
{
    /// <summary>
    /// Tokenize a string using the character tokenizer
    /// </summary>
    public static int[] Tokenize(this CharacterTokenizer tokenizer, string text)
    {
        return tokenizer.Encode(text.AsSpan());
    }

    /// <summary>
    /// Detokenize token IDs back to a string
    /// </summary>
    public static string Detokenize(this CharacterTokenizer tokenizer, int[] tokens)
    {
        return tokenizer.Decode(tokens.AsSpan());
    }

    /// <summary>
    /// Create a tokenizer from text data (convenience method)
    /// </summary>
    public static CharacterTokenizer CreateFromText(string text)
    {
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));

        var tokenizer = new CharacterTokenizer();
        tokenizer.Fit(text.AsSpan());
        return tokenizer;
    }

    /// <summary>
    /// Get a summary of what characters are in the text
    /// </summary>
    public static string GetTextSummary(this CharacterTokenizer tokenizer, ReadOnlySpan<char> text)
    {
        if (!tokenizer.IsFitted)
            return "Tokenizer not fitted";

        var knownChars = 0;
        var unknownChars = 0;
        var unknownCharSet = new HashSet<char>();

        foreach (char c in text)
        {
            if (tokenizer.ContainsCharacter(c))
            {
                knownChars++;
            }
            else
            {
                unknownChars++;
                unknownCharSet.Add(c);
            }
        }

        var totalChars = knownChars + unknownChars;
        var coverage = totalChars > 0 ? (double)knownChars / totalChars * 100 : 0;

        var summary = $"Text analysis: {totalChars} characters, {coverage:F1}% coverage ({knownChars} known, {unknownChars} unknown)";

        if (unknownChars > 0)
        {
            var unknownList = string.Join(", ", unknownCharSet.Take(10).Select(c => $"'{c}'"));
            if (unknownCharSet.Count > 10)
                unknownList += $" and {unknownCharSet.Count - 10} more";
            summary += $"\nUnknown characters: {unknownList}";
        }

        return summary;
    }

    /// <summary>
    /// Encode text with detailed error information if any characters are unknown
    /// </summary>
    public static TokenizationResult TryEncode(this CharacterTokenizer tokenizer, ReadOnlySpan<char> text)
    {
        if (!tokenizer.IsFitted)
        {
            return new TokenizationResult(
                Success: false,
                Tokens: Array.Empty<int>(),
                ErrorMessage: "Tokenizer must be fitted before encoding"
            );
        }

        var unknownChars = new List<(char Character, int Position)>();
        var tokens = new int[text.Length];
        bool success = true;

        for (int i = 0; i < text.Length; i++)
        {
            if (tokenizer.ContainsCharacter(text[i]))
            {
                tokens[i] = tokenizer.GetTokenId(text[i]);
            }
            else
            {
                unknownChars.Add((text[i], i));
                success = false;
            }
        }

        if (success)
        {
            return new TokenizationResult(
                Success: true,
                Tokens: tokens,
                ErrorMessage: null
            );
        }
        else
        {
            var errorMsg = $"Found {unknownChars.Count} unknown characters: " +
                          string.Join(", ", unknownChars.Take(5).Select(x => $"'{x.Character}' at position {x.Position}"));
            if (unknownChars.Count > 5)
                errorMsg += $" and {unknownChars.Count - 5} more";

            return new TokenizationResult(
                Success: false,
                Tokens: Array.Empty<int>(),
                ErrorMessage: errorMsg
            );
        }
    }
}

/// <summary>
/// Result of a tokenization attempt
/// </summary>
/// <param name="Success">Whether tokenization was successful</param>
/// <param name="Tokens">Token IDs (empty if failed)</param>
/// <param name="ErrorMessage">Error description (null if successful)</param>
public record TokenizationResult(
    bool Success,
    int[] Tokens,
    string? ErrorMessage
);