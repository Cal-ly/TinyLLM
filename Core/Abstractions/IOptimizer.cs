using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// Represents an optimization algorithm for updating model weights
/// </summary>
public interface IOptimizer
{
    /// <summary>
    /// The current learning rate
    /// </summary>
    float LearningRate { get; set; }

    /// <summary>
    /// Number of optimization steps taken
    /// </summary>
    int Step { get; }

    /// <summary>
    /// Update weights for a specific parameter
    /// </summary>
    /// <param name="parameterName">Unique name for this parameter group</param>
    /// <param name="weights">Current weights (will be modified in-place)</param>
    /// <param name="gradients">Gradients for these weights</param>
    void UpdateWeights(string parameterName, Span<float> weights, ReadOnlySpan<float> gradients);

    /// <summary>
    /// Reset optimizer state (e.g., momentum terms)
    /// </summary>
    void Reset();

    /// <summary>
    /// Get optimizer state for checkpointing
    /// </summary>
    OptimizerState GetState();

    /// <summary>
    /// Load optimizer state from checkpoint
    /// </summary>
    void LoadState(OptimizerState state);
}
