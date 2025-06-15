using Core.Abstractions;
using DataPipeline.Abstractions;

namespace DataPipeline.Loading;
/// <summary>
/// Represents a tokenized text dataset ready for training
/// </summary>
public sealed class TextDataset
{
    private readonly int[] _tokens;
    private readonly ITokenizer _tokenizer;
    private readonly string _sourceFile;
    private readonly DatasetMetadata _metadata;

    public TextDataset(
        int[] tokens,
        ITokenizer tokenizer,
        string sourceFile,
        DatasetMetadata metadata)
    {
        _tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _sourceFile = sourceFile ?? throw new ArgumentNullException(nameof(sourceFile));
        _metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
    }

    /// <summary>
    /// Total number of tokens in the dataset
    /// </summary>
    public int TokenCount => _tokens.Length;

    /// <summary>
    /// The tokenizer used to create this dataset
    /// </summary>
    public ITokenizer Tokenizer => _tokenizer;

    /// <summary>
    /// Source file path
    /// </summary>
    public string SourceFile => _sourceFile;

    /// <summary>
    /// Metadata about the dataset
    /// </summary>
    public DatasetMetadata Metadata => _metadata;

    /// <summary>
    /// Get all tokens as a read-only span
    /// </summary>
    public ReadOnlySpan<int> GetTokens() => _tokens.AsSpan();

    /// <summary>
    /// Get a slice of tokens
    /// </summary>
    /// <param name="start">Starting index</param>
    /// <param name="length">Number of tokens to get</param>
    /// <returns>Slice of tokens</returns>
    public ReadOnlySpan<int> GetTokens(int start, int length)
    {
        if (start < 0 || start >= _tokens.Length)
            throw new ArgumentOutOfRangeException(nameof(start));
        if (length < 0 || (start + length) > _tokens.Length)
            throw new ArgumentOutOfRangeException(nameof(length));

        return _tokens.AsSpan(start, length);
    }

    /// <summary>
    /// Create training batches from this dataset
    /// </summary>
    /// <param name="batchSize">Number of sequences per batch</param>
    /// <param name="sequenceLength">Length of each sequence</param>
    /// <param name="shuffle">Whether to shuffle the data</param>
    /// <param name="seed">Random seed for shuffling</param>
    /// <returns>Enumerable of training batches</returns>
    public IEnumerable<TrainingBatch> CreateBatches(
        int batchSize,
        int sequenceLength,
        bool shuffle = true,
        int? seed = null)
    {
        return BatchGenerator.CreateBatches(this, batchSize, sequenceLength, shuffle, seed);
    }

    /// <summary>
    /// Split dataset into training and validation sets
    /// </summary>
    /// <param name="trainSplit">Fraction of data for training (0.0 to 1.0)</param>
    /// <param name="seed">Random seed for splitting</param>
    /// <returns>Training and validation datasets</returns>
    public (TextDataset Training, TextDataset Validation) Split(float trainSplit = 0.9f, int? seed = null)
    {
        if (trainSplit <= 0f || trainSplit >= 1f)
            throw new ArgumentException("Train split must be between 0 and 1", nameof(trainSplit));

        var splitter = new DatasetSplitter(seed);
        return splitter.Split(this, trainSplit);
    }

    /// <summary>
    /// Get a sample of text from the dataset (for debugging)
    /// </summary>
    /// <param name="start">Starting token index</param>
    /// <param name="length">Number of tokens to include</param>
    /// <returns>Decoded text sample</returns>
    public string GetTextSample(int start = 0, int length = 100)
    {
        if (start < 0 || start >= _tokens.Length)
            throw new ArgumentOutOfRangeException(nameof(start));

        var actualLength = Math.Min(length, _tokens.Length - start);
        var sample = GetTokens(start, actualLength);
        return _tokenizer.Decode(sample);
    }

    /// <summary>
    /// Get statistics about the dataset
    /// </summary>
    public DatasetStatistics GetStatistics()
    {
        if (_tokens.Length == 0)
        {
            return new DatasetStatistics(
                TotalTokens: 0,
                UniqueTokens: 0,
                VocabularySize: _tokenizer.VocabularySize,
                MostFrequentTokens: Array.Empty<(int TokenId, int Count)>(),
                TokenFrequencies: new Dictionary<int, int>()
            );
        }

        // Count token frequencies
        var frequencies = new Dictionary<int, int>();
        foreach (int token in _tokens)
        {
            frequencies[token] = frequencies.GetValueOrDefault(token, 0) + 1;
        }

        var mostFrequent = frequencies
            .OrderByDescending(kvp => kvp.Value)
            .Take(10)
            .Select(kvp => (kvp.Key, kvp.Value))
            .ToArray();

        return new DatasetStatistics(
            TotalTokens: _tokens.Length,
            UniqueTokens: frequencies.Count,
            VocabularySize: _tokenizer.VocabularySize,
            MostFrequentTokens: mostFrequent,
            TokenFrequencies: frequencies
        );
    }
}
