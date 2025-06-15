using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using Core.Mathematics;

namespace Core.Models;
/// <summary>
/// Output layer that projects hidden states to vocabulary logits
/// </summary>
public sealed class OutputLayer : ILayer
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private float[,] _weights;     // [input_dim, output_dim]
    private float[]? _bias;        // [output_dim] - optional

    // Cache for backward pass
    private float[] _lastInput = Array.Empty<float>();

    public string Name => "OutputProjection";
    public int ParameterCount => (_inputDim * _outputDim) + (_bias?.Length ?? 0);

    public OutputLayer(int inputDim, int outputDim, bool useBias = false)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _weights = new float[inputDim, outputDim];
        _bias = useBias ? new float[outputDim] : null;
    }

    /// <summary>
    /// Forward pass: project hidden state to vocabulary logits
    /// </summary>
    /// <param name="input">Hidden state vector [input_dim]</param>
    /// <returns>Logits [output_dim]</returns>
    public float[] Forward(float[] input)
    {
        if (input.Length != _inputDim)
            throw new ArgumentException($"Input dimension {input.Length} doesn't match expected {_inputDim}");

        _lastInput = (float[])input.Clone();
        var output = new float[_outputDim];

        // Matrix-vector multiplication: output = weights^T * input + bias
        for (int out_idx = 0; out_idx < _outputDim; out_idx++)
        {
            float sum = _bias?[out_idx] ?? 0f; // Add bias if present
            for (int in_idx = 0; in_idx < _inputDim; in_idx++)
            {
                sum += _weights[in_idx, out_idx] * input[in_idx];
            }
            output[out_idx] = sum;
        }

        return output;
    }

    /// <summary>
    /// Forward pass for ILayer interface (matrix input)
    /// </summary>
    public float[,] Forward(float[,] input)
    {
        // Apply to last token only (for language modeling)
        int seqLength = input.GetLength(0);
        var lastToken = new float[_inputDim];
        for (int i = 0; i < _inputDim; i++)
        {
            lastToken[i] = input[seqLength - 1, i];
        }

        var logits = Forward(lastToken);
        var result = new float[1, _outputDim];
        for (int i = 0; i < _outputDim; i++)
        {
            result[0, i] = logits[i];
        }
        return result;
    }

    /// <summary>
    /// Backward pass through output layer
    /// </summary>
    /// <param name="outputGradients">Gradients from loss [output_dim]</param>
    /// <returns>Gradients for weights and input</returns>
    public LayerGradients Backward(float[] outputGradients)
    {
        if (outputGradients.Length != _outputDim)
            throw new ArgumentException($"Output gradient dimension {outputGradients.Length} doesn't match expected {_outputDim}");

        // Gradients for weights: outer product of input and output gradients
        var weightGradients = new float[_inputDim, _outputDim];
        for (int in_idx = 0; in_idx < _inputDim; in_idx++)
        {
            for (int out_idx = 0; out_idx < _outputDim; out_idx++)
            {
                weightGradients[in_idx, out_idx] = _lastInput[in_idx] * outputGradients[out_idx];
            }
        }

        // Gradients for input: weights * output_gradients
        var inputGradients = new float[_inputDim];
        for (int in_idx = 0; in_idx < _inputDim; in_idx++)
        {
            float sum = 0f;
            for (int out_idx = 0; out_idx < _outputDim; out_idx++)
            {
                sum += _weights[in_idx, out_idx] * outputGradients[out_idx];
            }
            inputGradients[in_idx] = sum;
        }

        // Bias gradients (if bias is used)
        float[]? biasGradients = null;
        if (_bias != null)
        {
            biasGradients = (float[])outputGradients.Clone();
        }

        return new LayerGradients(
            WeightGradients: weightGradients,
            InputGradients: inputGradients.Unflatten(1, _inputDim), // Convert to 2D for consistency
            BiasGradients: biasGradients
        );
    }

    /// <summary>
    /// Backward pass for ILayer interface (matrix gradients)
    /// </summary>
    public LayerGradients Backward(float[,] outputGradients)
    {
        // Extract gradients for the single output row
        var singleGrad = new float[_outputDim];
        for (int i = 0; i < _outputDim; i++)
        {
            singleGrad[i] = outputGradients[0, i];
        }
        return Backward(singleGrad);
    }

    /// <summary>
    /// Apply gradients to weights and bias
    /// </summary>
    public void ApplyGradients(ReadOnlySpan<float> weightGradients, ReadOnlySpan<float> biasGradients, IOptimizer optimizer)
    {
        // Update weights
        var flatWeights = _weights.Flatten();
        optimizer.UpdateWeights($"{Name}_weights", flatWeights, weightGradients);
        _weights = flatWeights.Unflatten(_inputDim, _outputDim);

        // Update bias if present
        if (_bias != null && biasGradients.Length > 0)
        {
            optimizer.UpdateWeights($"{Name}_bias", _bias, biasGradients);
        }
    }

    /// <summary>
    /// Update weights using computed gradients
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        ApplyGradients(
            gradients.WeightGradients.Flatten(),
            gradients.BiasGradients ?? Array.Empty<float>(),
            optimizer
        );
    }

    /// <summary>
    /// Initialize output layer weights
    /// </summary>
    public void InitializeWeights(Random random, WeightInitializationEnum initType = WeightInitializationEnum.Xavier)
    {
        var flatWeights = _weights.Flatten();
        WeightInitialization.Initialize(initType, flatWeights, _inputDim, _outputDim, random);
        _weights = flatWeights.Unflatten(_inputDim, _outputDim);

        // Initialize bias to zero if present
        if (_bias != null)
        {
            Array.Fill(_bias, 0f);
        }
    }

    /// <summary>
    /// Initialize weights (ILayer interface)
    /// </summary>
    public void InitializeWeights(Random random)
    {
        InitializeWeights(random, WeightInitializationEnum.Xavier);
    }

    /// <summary>
    /// Get current parameters
    /// </summary>
    public (float[,] weights, float[]? bias) GetParameters()
    {
        return ((float[,])_weights.Clone(), _bias != null ? (float[])_bias.Clone() : null);
    }

    /// <summary>
    /// Load parameters
    /// </summary>
    public void LoadParameters(float[,] weights, float[]? bias)
    {
        if (weights.GetLength(0) != _inputDim || weights.GetLength(1) != _outputDim)
            throw new ArgumentException($"Weight dimensions mismatch: expected [{_inputDim}, {_outputDim}], got [{weights.GetLength(0)}, {weights.GetLength(1)}]");

        _weights = (float[,])weights.Clone();

        if (bias != null)
        {
            if (bias.Length != _outputDim)
                throw new ArgumentException($"Bias dimension mismatch: expected {_outputDim}, got {bias.Length}");
            _bias = (float[])bias.Clone();
        }
    }

    public LayerState GetState()
    {
        var weights = new Dictionary<string, float[]>
        {
            ["weights"] = _weights.Flatten()
        };

        if (_bias != null)
        {
            weights["bias"] = _bias;
        }

        return new LayerState(
            LayerType: "OutputProjection",
            Weights: weights,
            Metadata: new Dictionary<string, object>
            {
                ["input_dim"] = _inputDim,
                ["output_dim"] = _outputDim,
                ["use_bias"] = _bias != null
            }
        );
    }

    public void LoadState(LayerState state)
    {
        if (state.LayerType != "OutputProjection")
            throw new ArgumentException($"Expected OutputProjection layer, got {state.LayerType}");

        _weights = state.Weights["weights"].Unflatten(_inputDim, _outputDim);

        if (state.Weights.TryGetValue("bias", out var bias))
        {
            _bias = bias;
        }
    }
}
