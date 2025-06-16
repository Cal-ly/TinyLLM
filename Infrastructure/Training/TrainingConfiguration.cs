using Core.Abstractions;
using Core.Optimizers;
using Infrastructure.Loading;
using Infrastructure.Tokenization;

namespace Infrastructure.Training;

/// <summary>
/// Configuration for training a language model
/// </summary>
public class TrainingConfiguration
{
    public required string Name { get; init; }
    public required string DatasetPath { get; init; }
    public required string OutputDirectory { get; init; }

    // Model configuration
    public required ModelConfiguration Model { get; set; }

    // Optimizer configuration
    public required OptimizerConfiguration Optimizer { get; init; }

    // Training hyperparameters
    public int NumEpochs { get; init; } = 10;
    public int BatchSize { get; init; } = 32;
    public int SequenceLength { get; init; } = 128;
    public float ValidationSplit { get; init; } = 0.1f;

    // Logging and checkpointing
    public int LoggingFrequency { get; init; } = 100;
    public int CheckpointFrequency { get; init; } = 1;
    public int ValidationFrequency { get; init; } = 1;
    public int SampleGenerationFrequency { get; init; } = 1;
    public float SampleTemperature { get; init; } = 0.7f;
    public int MaxValidationBatches { get; init; } = 100;

    // Early stopping
    public int EarlyStoppingPatience { get; init; } = 5;

    // Other
    public int? RandomSeed { get; init; } = 42;
    public string? ResumeFromCheckpoint { get; init; }
    public string[] SamplePrompts { get; init; } = Array.Empty<string>();
    public bool MetricsCsvEnabled { get; init; } = false;

    public int GetTotalSteps()
    {
        // This is an approximation - actual steps depend on dataset size
        return NumEpochs * 1000; // Placeholder
    }
}

/// <summary>
/// Configuration for model evaluation
/// </summary>
public class EvaluationConfiguration
{
    public int BatchSize { get; init; } = 32;
    public int MaxBatches { get; init; } = int.MaxValue;
    public int SequenceLength { get; init; } = 128;
}

/// <summary>
/// Context containing all components needed for training
/// </summary>
public class TrainingContext
{
    public required ILanguageModel Model { get; init; }
    public required IOptimizer Optimizer { get; init; }
    public required LearningRateScheduler Scheduler { get; init; }
    public required LossComputer LossComputer { get; init; }
    public required TextDataset TrainDataset { get; init; }
    public TextDataset? ValidationDataset { get; init; }
    public required TrainingConfiguration Configuration { get; init; }
    public required CheckpointManager CheckpointManager { get; init; }
    public required ValidationRunner ValidationRunner { get; init; }
}

/// <summary>
/// Result of a training run
/// </summary>
public record TrainingResult(
    bool Success,
    int TotalEpochs,
    int TotalSteps,
    float BestValidationLoss,
    IReadOnlyList<EpochMetrics> TrainingHistory,
    ModelState FinalModelState
);

/// <summary>
/// Metrics for a single epoch
/// </summary>
public record EpochMetrics(
    int Epoch,
    float AverageLoss,
    int TotalSteps,
    float FinalLearningRate,
    float GradientNorm
);

/// <summary>
/// Validation metrics
/// </summary>
public record ValidationMetrics(
    float Loss,
    float Perplexity,
    float Accuracy
);

/// <summary>
/// Result of a single training step
/// </summary>
public record TrainingStepResult(
    float Loss,
    float GradientNorm
);

/// <summary>
/// Result of model evaluation
/// </summary>
public record EvaluationResult(
    float AverageLoss,
    float Perplexity,
    float Accuracy,
    int TotalBatches,
    TimeSpan EvaluationTime
);

/// <summary>
/// Model checkpoint data
/// </summary>
public record Checkpoint(
    ModelConfiguration Configuration,
    ModelState ModelState,
    OptimizerState OptimizerState,
    TokenizerState? TokenizerState,
    int Epoch,
    float Loss,
    DateTime SavedAt
);