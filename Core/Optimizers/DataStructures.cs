using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// Statistics about array values (for debugging optimizer state)
/// </summary>
public record ArrayStatistics(
    float Mean,
    float StandardDeviation,
    float Min,
    float Max
);

/// <summary>
/// Statistics for a specific parameter in an optimizer
/// </summary>
public record ParameterStatistics(
    float MomentumMean,
    float MomentumStd,
    float MomentumMax,
    float SecondMomentMean,
    float SecondMomentStd,
    float SecondMomentMax
);

/// <summary>
/// Comprehensive statistics about an optimizer's state
/// </summary>
public record OptimizerStatistics(
    OptimizerTypeEnum OptimizerType,
    int Step,
    float LearningRate,
    IReadOnlyDictionary<string, ParameterStatistics> ParameterStats
)
{
    /// <summary>
    /// Get overall gradient statistics across all parameters
    /// </summary>
    public (float meanMomentum, float maxMomentum, float meanSecondMoment) GetOverallStatistics()
    {
        if (ParameterStats.Count == 0)
            return (0f, 0f, 0f);

        float totalMomentum = 0f;
        float maxMomentum = float.NegativeInfinity;
        float totalSecondMoment = 0f;

        foreach (var stats in ParameterStats.Values)
        {
            totalMomentum += stats.MomentumMean;
            if (stats.MomentumMax > maxMomentum)
                maxMomentum = stats.MomentumMax;
            totalSecondMoment += stats.SecondMomentMean;
        }

        return (
            meanMomentum: totalMomentum / ParameterStats.Count,
            maxMomentum: maxMomentum,
            meanSecondMoment: totalSecondMoment / ParameterStats.Count
        );
    }
}