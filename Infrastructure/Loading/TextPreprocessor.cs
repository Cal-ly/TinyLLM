using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infrastructure.Loading;
/// <summary>
/// Handles text preprocessing and cleaning
/// </summary>
public sealed class TextPreprocessor
{
    /// <summary>
    /// Process text according to the specified options
    /// </summary>
    /// <param name="text">Raw input text</param>
    /// <param name="options">Preprocessing options</param>
    /// <returns>Processed text</returns>
    public string Process(string text, PreprocessingOptions options)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        var processed = text;

        // Apply transformations in order
        if (options.RemoveExtraWhitespace)
            processed = RemoveExtraWhitespace(processed);

        if (options.NormalizeLineEndings)
            processed = NormalizeLineEndings(processed);

        if (options.RemoveControlCharacters)
            processed = RemoveControlCharacters(processed);

        if (options.ToLowercase)
            processed = processed.ToLowerInvariant();

        if (options.TrimWhitespace)
            processed = processed.Trim();

        if (options.MinimumLength > 0 && processed.Length < options.MinimumLength)
            return string.Empty;

        if (options.MaximumLength > 0 && processed.Length > options.MaximumLength)
            processed = processed[..options.MaximumLength];

        return processed;
    }

    /// <summary>
    /// Remove extra whitespace while preserving line structure
    /// </summary>
    private static string RemoveExtraWhitespace(string text)
    {
        // Replace multiple spaces with single space, but preserve line breaks
        var lines = text.Split('\n');
        var processedLines = lines.Select(line =>
            string.Join(' ', line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        );
        return string.Join('\n', processedLines);
    }

    /// <summary>
    /// Normalize line endings to \n
    /// </summary>
    private static string NormalizeLineEndings(string text)
    {
        return text.Replace("\r\n", "\n").Replace("\r", "\n");
    }

    /// <summary>
    /// Remove control characters except for common whitespace
    /// </summary>
    private static string RemoveControlCharacters(string text)
    {
        var result = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            // Keep printable characters and common whitespace
            if (!char.IsControl(c) || c == '\n' || c == '\t')
            {
                result.Append(c);
            }
        }
        return result.ToString();
    }
}
