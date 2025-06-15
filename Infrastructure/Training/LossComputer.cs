using Core.Mathematics;

namespace Infrastructure.Training;

/// <summary>
/// Computes loss for language modeling training
/// </summary>
public sealed class LossComputer
{
    /// <summary>
    /// Compute cross-entropy loss for next token prediction
    /// </summary>
    public float ComputeLoss(ReadOnlySpan<float> logits, int targetToken)
    {
        if (targetToken < 0 || targetToken >= logits.Length)
            throw new ArgumentException($"Target token {targetToken} out of range [0, {logits.Length - 1}]");

        // Convert logits to probabilities
        var probabilities = new float[logits.Length];
        NumericalFunctions.Softmax(logits, probabilities);

        // Cross-entropy loss: -log(p(target))
        var targetProbability = Math.Max(probabilities[targetToken], 1e-7f); // Avoid log(0)
        return -MathF.Log(targetProbability);
    }

    /// <summary>
    /// Compute loss for a sequence (average over all positions)
    /// </summary>
    public float ComputeSequenceLoss(ReadOnlySpan<float> logits, ReadOnlySpan<int> targets, int vocabSize)
    {
        if (logits.Length != targets.Length * vocabSize)
            throw new ArgumentException("Logits and targets dimension mismatch");

        float totalLoss = 0f;

        for (int i = 0; i < targets.Length; i++)
        {
            var positionLogits = logits.Slice(i * vocabSize, vocabSize);
            var loss = ComputeLoss(positionLogits, targets[i]);
            totalLoss += loss;
        }

        return totalLoss / targets.Length;
    }

    /// <summary>
    /// Compute perplexity from loss
    /// </summary>
    public float ComputePerplexity(float averageLoss)
    {
        return MathF.Exp(averageLoss);
    }

    /// <summary>
    /// Compute accuracy (fraction of correct predictions)
    /// </summary>
    public float ComputeAccuracy(ReadOnlySpan<float> logits, ReadOnlySpan<int> targets, int vocabSize)
    {
        if (logits.Length != targets.Length * vocabSize)
            throw new ArgumentException("Logits and targets dimension mismatch");

        int correct = 0;

        for (int i = 0; i < targets.Length; i++)
        {
            var positionLogits = logits.Slice(i * vocabSize, vocabSize);
            var predictedToken = GetMaxIndex(positionLogits);

            if (predictedToken == targets[i])
                correct++;
        }

        return (float)correct / targets.Length;
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