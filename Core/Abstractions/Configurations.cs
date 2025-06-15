namespace Core.Abstractions;
/// <summary>
/// Configuration for the overall model architecture
/// </summary>
public record ModelConfiguration
{
    public required int EmbeddingDim { get; init; }
    public required int VocabularySize { get; init; }
    public required int ContextLength { get; init; }
    public required int NumLayers { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int FeedForwardDim { get; init; }
    public float DropoutRate { get; init; } = 0.1f;
    public bool UseLayerNorm { get; init; } = true;
    public bool UseResidualConnections { get; init; } = true;
    public ActivationTypeEnum ActivationFunction { get; init; } = ActivationTypeEnum.ReLU;
    public WeightInitializationEnum WeightInit { get; init; } = WeightInitializationEnum.Xavier;

    /// <summary>
    /// Validate configuration parameters
    /// </summary>
    public void Validate()
    {
        if (EmbeddingDim <= 0)
            throw new ArgumentException("EmbeddingDim must be positive");
        if (VocabularySize <= 0)
            throw new ArgumentException("VocabularySize must be positive");
        if (ContextLength <= 0)
            throw new ArgumentException("ContextLength must be positive");
        if (NumLayers <= 0)
            throw new ArgumentException("NumLayers must be positive");
        if (NumAttentionHeads <= 0)
            throw new ArgumentException("NumAttentionHeads must be positive");
        if (EmbeddingDim % NumAttentionHeads != 0)
            throw new ArgumentException("EmbeddingDim must be divisible by NumAttentionHeads");
        if (FeedForwardDim <= 0)
            throw new ArgumentException("FeedForwardDim must be positive");
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be in [0, 1)");
    }

    /// <summary>
    /// Calculate total number of parameters for this configuration
    /// </summary>
    public long CalculateParameterCount()
    {
        long embeddingParams = (long)VocabularySize * EmbeddingDim;
        long outputParams = (long)EmbeddingDim * VocabularySize;

        // Per transformer layer: attention + feed-forward
        long attentionParams = 4L * EmbeddingDim * EmbeddingDim; // Q, K, V, O matrices
        long feedForwardParams = 2L * EmbeddingDim * FeedForwardDim; // Up and down projections
        long layerNormParams = UseLayerNorm ? 4L * EmbeddingDim : 0; // 2 layer norms per layer

        long perLayerParams = attentionParams + feedForwardParams + layerNormParams;
        long totalLayerParams = NumLayers * perLayerParams;

        return embeddingParams + totalLayerParams + outputParams;
    }
}

/// <summary>
/// Configuration for optimization
/// </summary>
public record OptimizerConfiguration
{
    public required OptimizerTypeEnum Type { get; init; }
    public required float LearningRate { get; init; }
    public float Beta1 { get; init; } = 0.9f;
    public float Beta2 { get; init; } = 0.999f;
    public float WeightDecay { get; init; } = 0.01f;
    public float GradientClipNorm { get; init; } = 1.0f;
    public float Epsilon { get; init; } = 1e-8f;
    public LearningRateSchedule? Schedule { get; init; }
}

/// <summary>
/// Learning rate schedule configuration
/// </summary>
public record LearningRateSchedule
{
    public required ScheduleTypeEnum Type { get; init; }
    public int WarmupSteps { get; init; } = 0;
    public float MinLearningRate { get; init; } = 0.0f;
    public float DecayRate { get; init; } = 0.1f;
    public int DecaySteps { get; init; } = 1000;
}
