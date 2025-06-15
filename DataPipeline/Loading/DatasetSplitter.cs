using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPipeline.Loading;
/// <summary>
/// Handles splitting datasets into training and validation sets
/// </summary>
public sealed class DatasetSplitter
{
    private readonly Random _random;

    public DatasetSplitter(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Split a dataset into training and validation sets
    /// </summary>
    /// <param name="dataset">Dataset to split</param>
    /// <param name="trainSplit">Fraction for training (0.0 to 1.0)</param>
    /// <returns>Training and validation datasets</returns>
    public (TextDataset Training, TextDataset Validation) Split(TextDataset dataset, float trainSplit)
    {
        if (trainSplit <= 0f || trainSplit >= 1f)
            throw new ArgumentException("Train split must be between 0 and 1", nameof(trainSplit));

        var tokens = dataset.GetTokens().ToArray();
        var splitIndex = (int)(tokens.Length * trainSplit);

        // Ensure we don't create empty datasets
        splitIndex = Math.Max(1, Math.Min(splitIndex, tokens.Length - 1));

        var trainTokens = tokens[..splitIndex];
        var valTokens = tokens[splitIndex..];

        var trainMetadata = dataset.Metadata with
        {
            OriginalFileName = $"{dataset.Metadata.OriginalFileName}_train",
            ProcessedTextLength = trainTokens.Length
        };

        var valMetadata = dataset.Metadata with
        {
            OriginalFileName = $"{dataset.Metadata.OriginalFileName}_val",
            ProcessedTextLength = valTokens.Length
        };

        var trainDataset = new TextDataset(trainTokens, dataset.Tokenizer, dataset.SourceFile + "_train", trainMetadata);
        var valDataset = new TextDataset(valTokens, dataset.Tokenizer, dataset.SourceFile + "_val", valMetadata);

        return (trainDataset, valDataset);
    }
}
