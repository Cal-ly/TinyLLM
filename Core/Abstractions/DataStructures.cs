using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// Collection of gradients for all model parameters
/// </summary>
public class GradientCollection
{
    private readonly Dictionary<string, float[]> _gradients = new();

    public void Add(string parameterName, float[] gradients)
    {
        _gradients[parameterName] = gradients;
    }

    public ReadOnlySpan<float> GetGradients(string parameterName)
    {
        return _gradients.TryGetValue(parameterName, out var gradients)
            ? gradients.AsSpan()
            : ReadOnlySpan<float>.Empty;
    }

    public IEnumerable<string> ParameterNames => _gradients.Keys;

    public bool HasGradients(string parameterName) => _gradients.ContainsKey(parameterName);

    public void Clear() => _gradients.Clear();

    /// <summary>
    /// Apply gradient clipping to all gradients
    /// </summary>
    public void ClipGradients(float maxNorm)
    {
        foreach (var gradients in _gradients.Values)
        {
            // Manual loop for best performance, even though I love LINQ <3
            float sumOfSquares = 0f;
            for (int i = 0; i < gradients.Length; i++)
            {
                sumOfSquares += gradients[i] * gradients[i];
            }
            var norm = MathF.Sqrt(sumOfSquares);

            if (norm > maxNorm)
            {
                var scale = maxNorm / norm;
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] *= scale;
                }
            }
        }
    }
}

/// <summary>
/// Gradients for a single layer
/// </summary>
public record LayerGradients(
    float[,] WeightGradients,
    float[,] InputGradients,
    float[]? BiasGradients = null
);

/// <summary>
/// Serializable state of a model
/// </summary>
public record ModelState(
    ModelConfiguration Configuration,
    Dictionary<string, float[]> Parameters,
    int TrainingStep = 0,
    float LastLoss = 0.0f
);

/// <summary>
/// Serializable state of an optimizer
/// </summary>
public record OptimizerState(
    OptimizerTypeEnum Type,
    Dictionary<string, float[]> Moments,
    int Step = 0,
    float LearningRate = 0.001f
);

/// <summary>
/// Serializable state of a layer
/// </summary>
public record LayerState(
    string LayerType,
    Dictionary<string, float[]> Weights,
    Dictionary<string, object> Metadata
);
