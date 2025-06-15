using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// Represents a single layer in a neural network
/// </summary>
public interface ILayer
{
    /// <summary>
    /// Name of this layer (for debugging and gradient tracking)
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Number of parameters in this layer
    /// </summary>
    int ParameterCount { get; }

    /// <summary>
    /// Forward pass through this layer
    /// </summary>
    float[,] Forward(float[,] input);

    /// <summary>
    /// Backward pass through this layer
    /// </summary>
    /// <param name="outputGradients">Gradients flowing backward from next layer</param>
    /// <returns>Gradients for weights and input</returns>
    LayerGradients Backward(float[,] outputGradients);

    /// <summary>
    /// Update layer weights using computed gradients
    /// </summary>
    void UpdateWeights(LayerGradients gradients, IOptimizer optimizer);

    /// <summary>
    /// Initialize layer weights
    /// </summary>
    void InitializeWeights(Random random);

    /// <summary>
    /// Get layer state for serialization
    /// </summary>
    LayerState GetState();

    /// <summary>
    /// Load layer state from serialization
    /// </summary>
    void LoadState(LayerState state);
}
