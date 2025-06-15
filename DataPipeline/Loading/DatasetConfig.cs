using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPipeline.Loading;
/// <summary>
/// Options for text preprocessing
/// </summary>
public record PreprocessingOptions(
    bool RemoveExtraWhitespace = true,
    bool NormalizeLineEndings = true,
    bool RemoveControlCharacters = false,
    bool ToLowercase = false,
    bool TrimWhitespace = true,
    int MinimumLength = 0,
    int MaximumLength = 0
)
{
    public static PreprocessingOptions Default => new();

    public static PreprocessingOptions ForShakespeare => new(
        RemoveExtraWhitespace: true,
        NormalizeLineEndings: true,
        RemoveControlCharacters: true,
        ToLowercase: false, // Keep original capitalization
        TrimWhitespace: true,
        MinimumLength: 10
    );

    public static PreprocessingOptions ForCode => new(
        RemoveExtraWhitespace: false, // Preserve code formatting
        NormalizeLineEndings: true,
        RemoveControlCharacters: false,
        ToLowercase: false,
        TrimWhitespace: false, // Preserve indentation
        MinimumLength: 5
    );
}

/// <summary>
/// Metadata about a loaded dataset
/// </summary>
public record DatasetMetadata(
    string OriginalFileName,
    long OriginalFileSize,
    DateTime LoadedAt,
    PreprocessingOptions PreprocessingOptions,
    int OriginalTextLength,
    int ProcessedTextLength,
    double CompressionRatio
);

/// <summary>
/// Statistics about a dataset
/// </summary>
public record DatasetStatistics(
    int TotalTokens,
    int UniqueTokens,
    int VocabularySize,
    IReadOnlyList<(int TokenId, int Count)> MostFrequentTokens,
    IReadOnlyDictionary<int, int> TokenFrequencies
)
{
    /// <summary>
    /// Calculate vocabulary coverage (what fraction of vocab is actually used)
    /// </summary>
    public double VocabularyCoverage => VocabularySize > 0 ? (double)UniqueTokens / VocabularySize : 0.0;

    /// <summary>
    /// Calculate average token frequency
    /// </summary>
    public double AverageTokenFrequency => UniqueTokens > 0 ? (double)TotalTokens / UniqueTokens : 0.0;
};