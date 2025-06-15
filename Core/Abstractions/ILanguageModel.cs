using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// Represents a language model capable of training and inference
/// </summary>
public interface ILanguageModel
{
    /// <summary>
    /// The current configuration of this model
    /// </summary>
    ModelConfiguration Configuration { get; }

    /// <summary>
    /// Whether the model has been initialized with weights
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Initialize the model with the given configuration
    /// </summary>
    void Initialize(ModelConfiguration config);

    /// <summary>
    /// Perform forward pass through the model
    /// </summary>
    /// <param name="inputTokens">Input token sequence</param>
    /// <returns>Logits for next token prediction</returns>
    float[] Forward(ReadOnlySpan<int> inputTokens);

    /// <summary>
    /// Perform backward pass to compute gradients
    /// </summary>
    /// <param name="loss">Loss value from forward pass</param>
    /// <returns>Gradients for all model parameters</returns>
    GradientCollection Backward(float loss);

    /// <summary>
    /// Apply computed gradients using the given optimizer
    /// </summary>
    void ApplyGradients(GradientCollection gradients, IOptimizer optimizer);

    /// <summary>
    /// Get the current state of the model for serialization
    /// </summary>
    ModelState GetState();

    /// <summary>
    /// Load model state from serialized data
    /// </summary>
    void LoadState(ModelState state);

    /// <summary>
    /// Reset the model to initial state (useful for testing)
    /// </summary>
    void Reset();
}
