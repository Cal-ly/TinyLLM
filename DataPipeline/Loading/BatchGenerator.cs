using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using DataPipeline.Abstractions;

namespace DataPipeline.Loading;
/// <summary>
/// Generates training batches from text datasets
/// </summary>
public static class BatchGenerator
{
    /// <summary>
    /// Create training batches from a dataset
    /// </summary>
    /// <param name="dataset">Source dataset</param>
    /// <param name="batchSize">Number of sequences per batch</param>
    /// <param name="sequenceLength">Length of each sequence</param>
    /// <param name="shuffle">Whether to shuffle the data</param>
    /// <param name="seed">Random seed for shuffling</param>
    /// <returns>Enumerable of training batches</returns>
    public static IEnumerable<TrainingBatch> CreateBatches(
        TextDataset dataset,
        int batchSize,
        int sequenceLength,
        bool shuffle = true,
        int? seed = null)
    {
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        if (sequenceLength <= 0)
            throw new ArgumentException("Sequence length must be positive", nameof(sequenceLength));

        var tokens = dataset.GetTokens();
        if (tokens.Length < (sequenceLength + 1))
            throw new ArgumentException($"Dataset too small: {tokens.Length} tokens, need at least {sequenceLength + 1}");

        // Calculate how many sequences we can create
        var maxSequences = tokens.Length - sequenceLength;
        var numBatches = (maxSequences + batchSize - 1) / batchSize;

        // Create sequence start indices
        var indices = Enumerable.Range(0, maxSequences).ToArray();

        if (shuffle)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            Shuffle(indices, random);
        }

        // Generate batches
        for (int batchIndex = 0; batchIndex < numBatches; batchIndex++)
        {
            var batchStart = batchIndex * batchSize;
            var actualBatchSize = Math.Min(batchSize, maxSequences - batchStart);

            var inputTokens = new int[actualBatchSize, sequenceLength];
            var targetTokens = new int[actualBatchSize, sequenceLength];

            for (int i = 0; i < actualBatchSize; i++)
            {
                var sequenceStart = indices[batchStart + i];

                // Input sequence: tokens[start..start+length]
                // Target sequence: tokens[start+1..start+length+1] (shifted by 1)
                for (int j = 0; j < sequenceLength; j++)
                {
                    inputTokens[i, j] = tokens[sequenceStart + j];
                    targetTokens[i, j] = tokens[sequenceStart + j + 1];
                }
            }

            yield return new TrainingBatch(inputTokens, targetTokens, actualBatchSize, sequenceLength);
        }
    }

    /// <summary>
    /// Fisher-Yates shuffle algorithm
    /// </summary>
    private static void Shuffle<T>(T[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}