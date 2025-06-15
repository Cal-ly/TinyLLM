using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DataPipeline.Abstractions;
using DataPipeline.Tokenization;

namespace DataPipeline.Loading;
/// <summary>
/// Loads text files and converts them into tokenized datasets
/// </summary>
public sealed class TextDataLoader(TextPreprocessor? preprocessor = null)
{
    private readonly TextPreprocessor _preprocessor = preprocessor ?? new TextPreprocessor();

    /// <summary>
    /// Load a text file and create a tokenized dataset
    /// </summary>
    /// <param name="filePath">Path to the text file</param>
    /// <param name="tokenizer">Tokenizer to use (if null, creates a new CharacterTokenizer)</param>
    /// <param name="preprocessingOptions">Text preprocessing options</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Loaded and tokenized dataset</returns>
    public async Task<TextDataset> LoadAsync(
        string filePath,
        ITokenizer? tokenizer = null,
        PreprocessingOptions? preprocessingOptions = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"File not found: {filePath}");

        var options = preprocessingOptions ?? PreprocessingOptions.Default;

        // Read the file
        string rawText;
        try
        {
            rawText = await File.ReadAllTextAsync(filePath, cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to read file '{filePath}': {ex.Message}", ex);
        }

        if (string.IsNullOrEmpty(rawText))
            throw new InvalidOperationException($"File '{filePath}' is empty or contains no readable text");

        // Preprocess the text
        var preprocessedText = _preprocessor.Process(rawText, options);

        if (string.IsNullOrEmpty(preprocessedText))
            throw new InvalidOperationException($"File '{filePath}' contains no valid text after preprocessing");

        // Create or fit tokenizer
        var finalTokenizer = tokenizer;
        if (finalTokenizer == null)
        {
            finalTokenizer = new CharacterTokenizer();
            finalTokenizer.Fit(preprocessedText.AsSpan());
        }
        else if (!finalTokenizer.IsFitted)
        {
            finalTokenizer.Fit(preprocessedText.AsSpan());
        }

        // Tokenize the text
        int[] tokens;
        try
        {
            tokens = finalTokenizer.Encode(preprocessedText.AsSpan());
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to tokenize text from '{filePath}': {ex.Message}", ex);
        }

        // Create metadata
        var fileInfo = new FileInfo(filePath);
        var metadata = new DatasetMetadata(
            OriginalFileName: Path.GetFileName(filePath),
            OriginalFileSize: fileInfo.Length,
            LoadedAt: DateTime.UtcNow,
            PreprocessingOptions: options,
            OriginalTextLength: rawText.Length,
            ProcessedTextLength: preprocessedText.Length,
            CompressionRatio: (double)preprocessedText.Length / rawText.Length
        );

        return new TextDataset(tokens, finalTokenizer, filePath, metadata);
    }

    /// <summary>
    /// Load multiple text files and combine them into a single dataset
    /// </summary>
    /// <param name="filePaths">Paths to text files</param>
    /// <param name="tokenizer">Shared tokenizer (if null, creates one from all files)</param>
    /// <param name="preprocessingOptions">Text preprocessing options</param>
    /// <param name="separator">Text to insert between files (default: newline)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Combined dataset</returns>
    public async Task<TextDataset> LoadMultipleAsync(
        IEnumerable<string> filePaths,
        ITokenizer? tokenizer = null,
        PreprocessingOptions? preprocessingOptions = null,
        string separator = "\n",
        CancellationToken cancellationToken = default)
    {
        var pathList = filePaths.ToList();
        if (pathList.Count == 0)
            throw new ArgumentException("At least one file path must be provided", nameof(filePaths));

        var options = preprocessingOptions ?? PreprocessingOptions.Default;
        var allTexts = new List<string>();
        var totalOriginalSize = 0L;

        // Read all files
        foreach (var filePath in pathList)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            var text = await File.ReadAllTextAsync(filePath, cancellationToken);
            var processed = _preprocessor.Process(text, options);

            if (!string.IsNullOrEmpty(processed))
            {
                allTexts.Add(processed);
                totalOriginalSize += new FileInfo(filePath).Length;
            }
        }

        if (allTexts.Count == 0)
            throw new InvalidOperationException("No valid text found in any of the provided files");

        // Combine texts
        var combinedText = string.Join(separator, allTexts);

        // Create or fit tokenizer
        var finalTokenizer = tokenizer;
        if (finalTokenizer == null)
        {
            finalTokenizer = new CharacterTokenizer();
            finalTokenizer.Fit(combinedText.AsSpan());
        }
        else if (!finalTokenizer.IsFitted)
        {
            finalTokenizer.Fit(combinedText.AsSpan());
        }

        // Tokenize
        var tokens = finalTokenizer.Encode(combinedText.AsSpan());

        // Create metadata
        var metadata = new DatasetMetadata(
            OriginalFileName: $"Combined_{pathList.Count}_files",
            OriginalFileSize: totalOriginalSize,
            LoadedAt: DateTime.UtcNow,
            PreprocessingOptions: options,
            OriginalTextLength: allTexts.Sum(t => t.Length + separator.Length) - separator.Length,
            ProcessedTextLength: combinedText.Length,
            CompressionRatio: 1.0 // Already processed
        );

        return new TextDataset(tokens, finalTokenizer, string.Join(";", pathList), metadata);
    }

    /// <summary>
    /// Create a dataset from raw text (useful for testing)
    /// </summary>
    /// <param name="text">Raw text content</param>
    /// <param name="tokenizer">Tokenizer to use</param>
    /// <param name="preprocessingOptions">Preprocessing options</param>
    /// <param name="name">Name for the dataset</param>
    /// <returns>Text dataset</returns>
    public TextDataset CreateFromText(
        string text,
        ITokenizer? tokenizer = null,
        PreprocessingOptions? preprocessingOptions = null,
        string name = "text_dataset")
    {
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));

        var options = preprocessingOptions ?? PreprocessingOptions.Default;
        var processedText = _preprocessor.Process(text, options);

        var finalTokenizer = tokenizer;
        if (finalTokenizer == null)
        {
            finalTokenizer = new CharacterTokenizer();
            finalTokenizer.Fit(processedText.AsSpan());
        }
        else if (!finalTokenizer.IsFitted)
        {
            finalTokenizer.Fit(processedText.AsSpan());
        }

        var tokens = finalTokenizer.Encode(processedText.AsSpan());

        var metadata = new DatasetMetadata(
            OriginalFileName: name,
            OriginalFileSize: text.Length * sizeof(char),
            LoadedAt: DateTime.UtcNow,
            PreprocessingOptions: options,
            OriginalTextLength: text.Length,
            ProcessedTextLength: processedText.Length,
            CompressionRatio: (double)processedText.Length / text.Length
        );

        return new TextDataset(tokens, finalTokenizer, name, metadata);
    }
}