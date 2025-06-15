using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Models;
/// <summary>
/// Factory class for creating common model configurations
/// </summary>
public static class ModelFactory
{
    /// <summary>
    /// Create a tiny model for testing and experimentation
    /// Perfect for Shakespeare character-level training
    /// </summary>
    public static ModelConfiguration CreateTinyModel(int vocabularySize)
    {
        return new ModelConfiguration
        {
            EmbeddingDim = 128,
            VocabularySize = vocabularySize,
            ContextLength = 128,
            NumLayers = 2,
            NumAttentionHeads = 4,
            FeedForwardDim = 512,
            DropoutRate = 0.1f,
            ActivationFunction = ActivationTypeEnum.GELU,
            WeightInit = WeightInitializationEnum.Xavier,
            UseLayerNorm = true,
            UseResidualConnections = true
        };
    }

    /// <summary>
    /// Create a small model suitable for educational purposes
    /// Good balance between capability and training time
    /// </summary>
    public static ModelConfiguration CreateSmallModel(int vocabularySize)
    {
        return new ModelConfiguration
        {
            EmbeddingDim = 256,
            VocabularySize = vocabularySize,
            ContextLength = 256,
            NumLayers = 4,
            NumAttentionHeads = 8,
            FeedForwardDim = 1024,
            DropoutRate = 0.1f,
            ActivationFunction = ActivationTypeEnum.GELU,
            WeightInit = WeightInitializationEnum.Xavier,
            UseLayerNorm = true,
            UseResidualConnections = true
        };
    }

    /// <summary>
    /// Create a medium model for more serious experimentation
    /// Suitable for Python code generation
    /// </summary>
    public static ModelConfiguration CreateMediumModel(int vocabularySize)
    {
        return new ModelConfiguration
        {
            EmbeddingDim = 384,
            VocabularySize = vocabularySize,
            ContextLength = 512,
            NumLayers = 6,
            NumAttentionHeads = 12,
            FeedForwardDim = 1536,
            DropoutRate = 0.15f,
            ActivationFunction = ActivationTypeEnum.GELU,
            WeightInit = WeightInitializationEnum.Xavier,
            UseLayerNorm = true,
            UseResidualConnections = true
        };
    }

    /// <summary>
    /// Create a custom model with specified parameters
    /// </summary>
    public static ModelConfiguration CreateCustomModel(
        int vocabularySize,
        int embeddingDim = 256,
        int contextLength = 256,
        int numLayers = 4,
        int numHeads = 8,
        ActivationTypeEnum activation = ActivationTypeEnum.GELU)
    {
        return new ModelConfiguration
        {
            EmbeddingDim = embeddingDim,
            VocabularySize = vocabularySize,
            ContextLength = contextLength,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            FeedForwardDim = embeddingDim * 4, // Standard 4x expansion
            DropoutRate = 0.1f,
            ActivationFunction = activation,
            WeightInit = WeightInitializationEnum.Xavier,
            UseLayerNorm = true,
            UseResidualConnections = true
        };
    }

    /// <summary>
    /// Get parameter count estimation for a configuration
    /// </summary>
    public static long EstimateParameterCount(ModelConfiguration config)
    {
        return config.CalculateParameterCount();
    }

    /// <summary>
    /// Get memory estimation in MB for a configuration
    /// </summary>
    public static float EstimateMemoryUsageMB(ModelConfiguration config)
    {
        long parameters = config.CalculateParameterCount();

        // Each parameter is 4 bytes (float32)
        // During training we need: weights + gradients + optimizer state (Adam: 2x momentum)
        // So roughly 4x the parameter count in bytes
        long bytesNeeded = parameters * 4 * 4; // 4 bytes per param, 4x for training overhead

        return bytesNeeded / (1024f * 1024f); // Convert to MB
    }

    /// <summary>
    /// Validate that a configuration will work on given hardware
    /// </summary>
    public static ValidationResult ValidateConfiguration(ModelConfiguration config, long availableMemoryMB)
    {
        var estimatedMemoryMB = EstimateMemoryUsageMB(config);
        var parameterCount = EstimateParameterCount(config);

        var warnings = new List<string>();
        var errors = new List<string>();

        // Memory check
        if (estimatedMemoryMB > availableMemoryMB)
        {
            errors.Add($"Estimated memory usage ({estimatedMemoryMB:F1} MB) exceeds available memory ({availableMemoryMB} MB)");
        }
        else if (estimatedMemoryMB > (availableMemoryMB * 0.8))
        {
            warnings.Add($"Memory usage ({estimatedMemoryMB:F1} MB) is close to limit ({availableMemoryMB} MB)");
        }

        // Configuration validation
        try
        {
            config.Validate();
        }
        catch (Exception ex)
        {
            errors.Add($"Configuration validation failed: {ex.Message}");
        }

        // Performance warnings
        if (config.ContextLength > 1024)
        {
            warnings.Add("Large context length may slow training significantly");
        }

        if (parameterCount > 100_000_000) // 100M parameters
        {
            warnings.Add("Large model may require significant training time");
        }

        return new ValidationResult(
            IsValid: errors.Count == 0,
            Errors: errors,
            Warnings: warnings,
            EstimatedParameterCount: parameterCount,
            EstimatedMemoryMB: estimatedMemoryMB
        );
    }
}

/// <summary>
/// Result of model configuration validation
/// </summary>
public record ValidationResult(
    bool IsValid,
    List<string> Errors,
    List<string> Warnings,
    long EstimatedParameterCount,
    float EstimatedMemoryMB
);