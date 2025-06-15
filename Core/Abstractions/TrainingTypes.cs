using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// A batch of training data
/// </summary>
public record TrainingBatch(
    int[,] InputTokens,   // [batch_size, sequence_length]
    int[,] TargetTokens,  // [batch_size, sequence_length] 
    int BatchSize,
    int SequenceLength
)
{
    public ReadOnlySpan<int> GetInputSequence(int batchIndex)
    {
        var input = new int[SequenceLength];
        for (int i = 0; i < SequenceLength; i++)
        {
            input[i] = InputTokens[batchIndex, i];
        }
        return input;
    }

    public ReadOnlySpan<int> GetTargetSequence(int batchIndex)
    {
        var target = new int[SequenceLength];
        for (int i = 0; i < SequenceLength; i++)
        {
            target[i] = TargetTokens[batchIndex, i];
        }
        return target;
    }
}

/// <summary>
/// Metrics collected during training
/// </summary>
public record TrainingMetrics(
    int Epoch,
    int Step,
    float Loss,
    float LearningRate,
    TimeSpan BatchTime,
    long MemoryUsage = 0
)
{
    public float TokensPerSecond(int batchSize, int sequenceLength)
    {
        return batchSize * sequenceLength / (float)BatchTime.TotalSeconds;
    }
}

/// <summary>
/// Result of a training step
/// </summary>
public record TrainingStepResult(
    float Loss,
    GradientCollection Gradients,
    TrainingMetrics Metrics
);
