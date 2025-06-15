using Core.Abstractions;
using Infrastructure.Loading;
using Infrastructure.Logging;

namespace Infrastructure.Training;

/// <summary>
/// Runs model evaluation/validation
/// </summary>
public class ValidationRunner
{
    private readonly ILogger _logger;
    private readonly LossComputer _lossComputer;

    public ValidationRunner(ILogger? logger = null)
    {
        _logger = logger ?? new ConsoleLogger("ValidationRunner");
        _lossComputer = new LossComputer();
    }

    public async Task<EvaluationResult> EvaluateAsync(
        ILanguageModel model,
        TextDataset dataset,
        EvaluationConfiguration config,
        CancellationToken cancellationToken = default)
    {
        await Task.Yield(); // Make async

        _logger.LogInformation("Starting model evaluation");
        var startTime = DateTime.UtcNow;

        float totalLoss = 0f;
        int totalCorrect = 0;
        int totalPredictions = 0;
        int batchCount = 0;

        var batches = dataset.CreateBatches(
            config.BatchSize,
            config.SequenceLength,
            shuffle: false); // Don't shuffle for evaluation

        foreach (var batch in batches.Take(config.MaxBatches))
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            // Process each sequence in the batch
            for (int i = 0; i < batch.BatchSize; i++)
            {
                var inputTokens = batch.GetInputSequence(i);
                var targetTokens = batch.GetTargetSequence(i);

                // Forward pass
                var logits = model.Forward(inputTokens);

                // Compute loss for the last token (next token prediction)
                var targetToken = targetTokens[targetTokens.Length - 1];
                var loss = _lossComputer.ComputeLoss(logits, targetToken);
                totalLoss += loss;

                // Check if prediction is correct
                var predictedToken = GetMaxIndex(logits);
                if (predictedToken == targetToken)
                {
                    totalCorrect++;
                }
                totalPredictions++;
            }

            batchCount++;

            if (batchCount % 10 == 0)
            {
                _logger.LogDebug("Evaluated {Count} batches", batchCount);
            }
        }

        var avgLoss = totalLoss / Math.Max(totalPredictions, 1);
        var perplexity = MathF.Exp(avgLoss);
        var accuracy = (float)totalCorrect / Math.Max(totalPredictions, 1);
        var duration = DateTime.UtcNow - startTime;

        _logger.LogInformation(
            "Evaluation complete - Loss: {Loss:F4}, Perplexity: {Perplexity:F2}, Accuracy: {Accuracy:P2}",
            avgLoss, perplexity, accuracy);

        return new EvaluationResult(
            AverageLoss: avgLoss,
            Perplexity: perplexity,
            Accuracy: accuracy,
            TotalBatches: batchCount,
            EvaluationTime: duration
        );
    }

    private static int GetMaxIndex(ReadOnlySpan<float> values)
    {
        int maxIndex = 0;
        float maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxValue)
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}