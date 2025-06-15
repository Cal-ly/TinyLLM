using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// Factory for creating optimizers with recommended configurations
/// </summary>
public static class OptimizerFactory
{
    /// <summary>
    /// Create an optimizer from configuration
    /// </summary>
    public static IOptimizer CreateFromConfiguration(OptimizerConfiguration config)
    {
        return config.Type switch
        {
            OptimizerTypeEnum.SGD => new SGDOptimizer(
                learningRate: config.LearningRate,
                momentum: config.Beta1, // Use Beta1 as momentum for SGD
                weightDecay: config.WeightDecay
            ),

            OptimizerTypeEnum.Adam => new AdamOptimizer(
                learningRate: config.LearningRate,
                beta1: config.Beta1,
                beta2: config.Beta2,
                epsilon: config.Epsilon,
                weightDecay: config.WeightDecay
            ),

            OptimizerTypeEnum.AdamW => new AdamWOptimizer(
                learningRate: config.LearningRate,
                beta1: config.Beta1,
                beta2: config.Beta2,
                epsilon: config.Epsilon,
                weightDecay: config.WeightDecay
            ),

            _ => throw new ArgumentException($"Unknown optimizer type: {config.Type}")
        };
    }

    /// <summary>
    /// Create Adam optimizer with recommended settings for transformers
    /// </summary>
    public static AdamOptimizer CreateAdam(float learningRate = 0.001f, float weightDecay = 0.01f)
    {
        return new AdamOptimizer(
            learningRate: learningRate,
            beta1: 0.9f,
            beta2: 0.999f,
            epsilon: 1e-8f,
            weightDecay: weightDecay
        );
    }

    /// <summary>
    /// Create AdamW optimizer with recommended settings for transformers
    /// </summary>
    public static AdamWOptimizer CreateAdamW(float learningRate = 0.001f, float weightDecay = 0.01f)
    {
        return new AdamWOptimizer(
            learningRate: learningRate,
            beta1: 0.9f,
            beta2: 0.999f,
            epsilon: 1e-8f,
            weightDecay: weightDecay
        );
    }

    /// <summary>
    /// Create SGD optimizer with momentum
    /// </summary>
    public static SGDOptimizer CreateSGD(float learningRate = 0.01f, float momentum = 0.9f, float weightDecay = 0.0f)
    {
        return new SGDOptimizer(
            learningRate: learningRate,
            momentum: momentum,
            weightDecay: weightDecay
        );
    }

    /// <summary>
    /// Get recommended optimizer configuration for different model sizes
    /// </summary>
    public static OptimizerConfiguration GetRecommendedConfiguration(string modelSize, long parameterCount)
    {
        return modelSize.ToLowerInvariant() switch
        {
            "tiny" => new OptimizerConfiguration
            {
                Type = OptimizerTypeEnum.Adam,
                LearningRate = 0.003f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                WeightDecay = 0.01f,
                GradientClipNorm = 1.0f,
                Schedule = new LearningRateSchedule
                {
                    Type = ScheduleTypeEnum.Cosine,
                    WarmupSteps = 500,
                    MinLearningRate = 1e-5f
                }
            },

            "small" => new OptimizerConfiguration
            {
                Type = OptimizerTypeEnum.AdamW,
                LearningRate = 0.001f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                WeightDecay = 0.01f,
                GradientClipNorm = 1.0f,
                Schedule = new LearningRateSchedule
                {
                    Type = ScheduleTypeEnum.Cosine,
                    WarmupSteps = 2000,
                    MinLearningRate = 1e-6f
                }
            },

            "medium" => new OptimizerConfiguration
            {
                Type = OptimizerTypeEnum.AdamW,
                LearningRate = 0.0005f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                WeightDecay = 0.02f,
                GradientClipNorm = 0.5f,
                Schedule = new LearningRateSchedule
                {
                    Type = ScheduleTypeEnum.Cosine,
                    WarmupSteps = 4000,
                    MinLearningRate = 1e-6f
                }
            },

            _ => throw new ArgumentException($"Unknown model size: {modelSize}. Use 'tiny', 'small', or 'medium'.")
        };
    }

    /// <summary>
    /// Create optimizer with learning rate scheduler
    /// </summary>
    public static (IOptimizer optimizer, LearningRateScheduler scheduler) CreateWithScheduler(OptimizerConfiguration config)
    {
        var optimizer = CreateFromConfiguration(config);
        var scheduler = LearningRateScheduler.FromConfiguration(config);

        return (optimizer, scheduler);
    }

    /// <summary>
    /// Validate optimizer configuration
    /// </summary>
    public static ValidationResult ValidateConfiguration(OptimizerConfiguration config)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Basic validation
        if (config.LearningRate <= 0f)
            errors.Add("Learning rate must be positive");

        if (config.Beta1 < 0f || config.Beta1 >= 1f)
            errors.Add("Beta1 must be in [0, 1)");

        if (config.Beta2 < 0f || config.Beta2 >= 1f)
            errors.Add("Beta2 must be in [0, 1)");

        if (config.WeightDecay < 0f)
            errors.Add("Weight decay must be non-negative");

        // Warnings for potentially problematic settings
        if (config.LearningRate > 0.01f)
            warnings.Add("High learning rate may cause instability");

        if (config.WeightDecay > 0.1f)
            warnings.Add("High weight decay may hurt model capacity");

        if (config.GradientClipNorm > 5.0f)
            warnings.Add("High gradient clip norm may not be effective");

        return new ValidationResult(
            IsValid: errors.Count == 0,
            Errors: errors,
            Warnings: warnings,
            EstimatedParameterCount: 0, // Not applicable
            EstimatedMemoryMB: 0f // Not applicable
        );
    }
}
