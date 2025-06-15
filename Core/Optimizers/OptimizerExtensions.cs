using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// Extension methods for easier optimizer integration
/// </summary>
public static class OptimizerExtensions
{
    /// <summary>
    /// Update learning rate and apply gradients in one call
    /// </summary>
    public static void UpdateWithScheduler(
        this IOptimizer optimizer,
        LearningRateScheduler scheduler,
        int step,
        GradientCollection gradients,
        float gradientClipNorm = 1.0f,
        int? maxSteps = null)
    {
        // Update learning rate
        optimizer.LearningRate = scheduler.GetLearningRate(step, maxSteps);

        // Clip gradients if specified
        if (gradientClipNorm > 0f)
        {
            GradientClipper.ClipByGlobalNorm(gradients, gradientClipNorm);
        }

        // Apply gradients to each parameter
        foreach (var paramName in gradients.ParameterNames)
        {
            // Note: This would need access to the actual model weights
            // In practice, this would be called from the training loop
            throw new NotImplementedException("This method should be called from the training context with access to model weights");
        }
    }

    /// <summary>
    /// Get comprehensive training statistics
    /// </summary>
    public static TrainingStatistics GetTrainingStatistics(
        this IOptimizer optimizer,
        GradientCollection gradients)
    {
        var optimizerStats = optimizer.GetStatistics();
        var gradientNorm = GradientClipper.ComputeGlobalNorm(gradients);

        return new TrainingStatistics(
            Step: optimizerStats.Step,
            LearningRate: optimizerStats.LearningRate,
            GradientNorm: gradientNorm,
            OptimizerStats: optimizerStats
        );
    }
}

/// <summary>
/// Comprehensive training statistics
/// </summary>
public record TrainingStatistics(
    int Step,
    float LearningRate,
    float GradientNorm,
    OptimizerStatistics OptimizerStats
);
