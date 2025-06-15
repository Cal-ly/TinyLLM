using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Models;
/// <summary>
/// Layer normalization - crucial for stable transformer training
/// </summary>
public sealed class LayerNormalization : ILayer
{
    private readonly int _normalizedShape;
    private readonly float _epsilon;

    private float[] _scale;  // Learnable scale parameter
    private float[] _bias;   // Learnable bias parameter

    // Cached for backward pass
    private float[,] _lastInput = new float[0, 0];
    private float[,] _lastNormalized = new float[0, 0];
    private float[] _lastMean = Array.Empty<float>();
    private float[] _lastInvStd = Array.Empty<float>();

    public string Name => "LayerNorm";
    public int ParameterCount => 2 * _normalizedShape;

    public LayerNormalization(int normalizedShape, float epsilon = 1e-6f)
    {
        _normalizedShape = normalizedShape;
        _epsilon = epsilon;
        _scale = new float[normalizedShape];
        _bias = new float[normalizedShape];
    }

    /// <summary>
    /// Forward pass: normalize along the last dimension
    /// </summary>
    public float[,] Forward(float[,] input)
    {
        int seqLength = input.GetLength(0);
        int features = input.GetLength(1);

        if (features != _normalizedShape)
            throw new ArgumentException($"Input feature size {features} doesn't match normalized shape {_normalizedShape}");

        _lastInput = (float[,])input.Clone();
        var output = new float[seqLength, features];
        _lastMean = new float[seqLength];
        _lastInvStd = new float[seqLength];

        // Normalize each sequence position independently
        for (int seq = 0; seq < seqLength; seq++)
        {
            // Calculate mean
            float mean = 0f;
            for (int feat = 0; feat < features; feat++)
            {
                mean += input[seq, feat];
            }
            mean /= features;
            _lastMean[seq] = mean;

            // Calculate variance
            float variance = 0f;
            for (int feat = 0; feat < features; feat++)
            {
                float diff = input[seq, feat] - mean;
                variance += (diff * diff);
            }
            variance /= features;

            // Calculate inverse standard deviation
            float invStd = 1f / MathF.Sqrt(variance + _epsilon);
            _lastInvStd[seq] = invStd;

            // Normalize and apply scale/bias
            for (int feat = 0; feat < features; feat++)
            {
                float normalized = (input[seq, feat] - mean) * invStd;
                output[seq, feat] = (normalized * _scale[feat]) + _bias[feat];
            }
        }

        _lastNormalized = output;
        return output;
    }

    /// <summary>
    /// Backward pass through layer normalization
    /// </summary>
    public LayerGradients Backward(float[,] outputGradients)
    {
        int seqLength = outputGradients.GetLength(0);
        int features = outputGradients.GetLength(1);

        var inputGradients = new float[seqLength, features];
        var scaleGradients = new float[features];
        var biasGradients = new float[features];

        for (int seq = 0; seq < seqLength; seq++)
        {
            float mean = _lastMean[seq];
            float invStd = _lastInvStd[seq];

            // Gradients for scale and bias
            for (int feat = 0; feat < features; feat++)
            {
                float normalized = (_lastInput[seq, feat] - mean) * invStd;
                scaleGradients[feat] += normalized * outputGradients[seq, feat];
                biasGradients[feat] += outputGradients[seq, feat];
            }

            // Gradients for input (simplified)
            for (int feat = 0; feat < features; feat++)
            {
                inputGradients[seq, feat] = _scale[feat] * outputGradients[seq, feat] * invStd;
            }
        }

        return new LayerGradients(
            WeightGradients: CombineScaleBiasGradients(scaleGradients, biasGradients), // Combine into 2D array
            InputGradients: inputGradients,
            BiasGradients: biasGradients // Bias gradients remain as 1D array
        );
    }

    /// <summary>
    /// Apply gradients to scale and bias parameters
    /// </summary>
    public void ApplyGradients(ReadOnlySpan<float> scaleGradients, ReadOnlySpan<float> biasGradients, IOptimizer optimizer)
    {
        optimizer.UpdateWeights($"{Name}_scale", _scale, scaleGradients);
        optimizer.UpdateWeights($"{Name}_bias", _bias, biasGradients);
    }

    /// <summary>
    /// Update weights using computed gradients
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        var (scaleGrads, biasGrads) = SplitScaleBiasGradients(gradients.WeightGradients);
        ApplyGradients(scaleGrads, biasGrads, optimizer);
    }

    /// <summary>
    /// Initialize layer norm parameters
    /// </summary>
    public void InitializeWeights(Random random)
    {
        // Initialize scale to 1 and bias to 0 (standard practice)
        Array.Fill(_scale, 1f);
        Array.Fill(_bias, 0f);
    }

    /// <summary>
    /// Get current parameters
    /// </summary>
    public (float[] scale, float[] bias) GetParameters()
    {
        return ((float[])_scale.Clone(), (float[])_bias.Clone());
    }

    /// <summary>
    /// Load parameters
    /// </summary>
    public void LoadParameters(float[] scale, float[] bias)
    {
        if (scale.Length != _normalizedShape || bias.Length != _normalizedShape)
            throw new ArgumentException($"Parameter size mismatch: expected {_normalizedShape}");

        _scale = (float[])scale.Clone();
        _bias = (float[])bias.Clone();
    }

    public LayerState GetState()
    {
        return new LayerState(
            LayerType: "LayerNormalization",
            Weights: new Dictionary<string, float[]>
            {
                ["scale"] = _scale,
                ["bias"] = _bias
            },
            Metadata: new Dictionary<string, object>
            {
                ["normalized_shape"] = _normalizedShape,
                ["epsilon"] = _epsilon
            }
        );
    }

    public void LoadState(LayerState state)
    {
        if (state.LayerType != "LayerNormalization")
            throw new ArgumentException($"Expected LayerNormalization layer, got {state.LayerType}");

        _scale = state.Weights["scale"];
        _bias = state.Weights["bias"];
    }

    // Helper methods
    private float[,] CombineScaleBiasGradients(float[] scaleGrads, float[] biasGrads)
    {
        var combined = new float[1, scaleGrads.Length + biasGrads.Length];
        for (int i = 0; i < scaleGrads.Length; i++)
            combined[0, i] = scaleGrads[i];
        for (int i = 0; i < biasGrads.Length; i++)
            combined[0, scaleGrads.Length + i] = biasGrads[i];
        return combined;
    }

    private (float[] scale, float[] bias) SplitScaleBiasGradients(float[,] combined)
    {
        var scale = new float[_normalizedShape];
        var bias = new float[_normalizedShape];

        for (int i = 0; i < _normalizedShape; i++)
        {
            scale[i] = combined[0, i];
            bias[i] = combined[0, _normalizedShape + i];
        }

        return (scale, bias);
    }
}