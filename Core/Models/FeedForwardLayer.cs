using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using Core.Mathematics;

namespace Core.Models;
/// <summary>
/// Feed-forward network (MLP) layer used in transformer blocks
/// </summary>
public sealed class FeedForwardLayer : ILayer
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly ActivationTypeEnum _activationType;
    private readonly float _dropoutRate;

    private float[,] _upProjection;   // [input_dim, hidden_dim]
    private float[,] _downProjection; // [hidden_dim, input_dim]
    private float[] _upBias;          // [hidden_dim]
    private float[] _downBias;        // [input_dim]

    // Cached values for backward pass
    private float[,] _lastInput = new float[0, 0];
    private float[,] _lastUpOutput = new float[0, 0];
    private float[,] _lastActivated = new float[0, 0];

    public string Name => "FeedForward";
    public int ParameterCount => (_inputDim * _hiddenDim) + (_hiddenDim * _inputDim) + _hiddenDim + _inputDim;

    public FeedForwardLayer(int inputDim, int hiddenDim, ActivationTypeEnum activationType, float dropoutRate = 0.1f)
    {
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _activationType = activationType;
        _dropoutRate = dropoutRate;

        _upProjection = new float[inputDim, hiddenDim];
        _downProjection = new float[hiddenDim, inputDim];
        _upBias = new float[hiddenDim];
        _downBias = new float[inputDim];
    }

    /// <summary>
    /// Forward pass: input -> up_projection -> activation -> down_projection
    /// </summary>
    public float[,] Forward(float[,] input)
    {
        int seqLength = input.GetLength(0);
        int inputDimCheck = input.GetLength(1);

        if (inputDimCheck != _inputDim)
            throw new ArgumentException($"Input dimension {inputDimCheck} doesn't match expected {_inputDim}");

        _lastInput = (float[,])input.Clone();

        // 1. Up projection: [seq_length, input_dim] -> [seq_length, hidden_dim]
        _lastUpOutput = new float[seqLength, _hiddenDim];

        for (int seq = 0; seq < seqLength; seq++)
        {
            for (int hidden = 0; hidden < _hiddenDim; hidden++)
            {
                float sum = _upBias[hidden]; // Add bias
                for (int inp = 0; inp < _inputDim; inp++)
                {
                    sum += input[seq, inp] * _upProjection[inp, hidden];
                }
                _lastUpOutput[seq, hidden] = sum;
            }
        }

        // 2. Apply activation function
        _lastActivated = new float[seqLength, _hiddenDim];
        for (int seq = 0; seq < seqLength; seq++)
        {
            var inputRow = new float[_hiddenDim];
            var outputRow = new float[_hiddenDim];

            for (int dim = 0; dim < _hiddenDim; dim++)
                inputRow[dim] = _lastUpOutput[seq, dim];

            ActivationFunctions.Apply(_activationType, inputRow, outputRow);

            for (int dim = 0; dim < _hiddenDim; dim++)
                _lastActivated[seq, dim] = outputRow[dim];
        }

        // 3. Down projection: [seq_length, hidden_dim] -> [seq_length, input_dim]
        var output = new float[seqLength, _inputDim];

        for (int seq = 0; seq < seqLength; seq++)
        {
            for (int inp = 0; inp < _inputDim; inp++)
            {
                float sum = _downBias[inp]; // Add bias
                for (int hidden = 0; hidden < _hiddenDim; hidden++)
                {
                    sum += _lastActivated[seq, hidden] * _downProjection[hidden, inp];
                }
                output[seq, inp] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass through feed-forward layer
    /// </summary>
    public LayerGradients Backward(float[,] outputGradients)
    {
        int seqLength = outputGradients.GetLength(0);

        // Gradients for down projection
        var downWeightGrads = new float[_hiddenDim, _inputDim];
        var downBiasGrads = new float[_inputDim];
        var activatedGrads = new float[seqLength, _hiddenDim];

        // Compute gradients for down projection weights and bias
        for (int hidden = 0; hidden < _hiddenDim; hidden++)
        {
            for (int inp = 0; inp < _inputDim; inp++)
            {
                float weightGrad = 0f;
                for (int seq = 0; seq < seqLength; seq++)
                {
                    weightGrad += _lastActivated[seq, hidden] * outputGradients[seq, inp];
                    activatedGrads[seq, hidden] += _downProjection[hidden, inp] * outputGradients[seq, inp];
                }
                downWeightGrads[hidden, inp] = weightGrad;
            }
        }

        // Bias gradients for down projection
        for (int inp = 0; inp < _inputDim; inp++)
        {
            for (int seq = 0; seq < seqLength; seq++)
            {
                downBiasGrads[inp] += outputGradients[seq, inp];
            }
        }

        // Gradients through activation function
        var upOutputGrads = new float[seqLength, _hiddenDim];
        for (int seq = 0; seq < seqLength; seq++)
        {
            var activationInput = new float[_hiddenDim];
            var activationDerivative = new float[_hiddenDim];

            for (int dim = 0; dim < _hiddenDim; dim++)
                activationInput[dim] = _lastUpOutput[seq, dim];

            ActivationFunctions.ApplyDerivative(_activationType, activationInput, activationDerivative);

            for (int dim = 0; dim < _hiddenDim; dim++)
                upOutputGrads[seq, dim] = activatedGrads[seq, dim] * activationDerivative[dim];
        }

        // Gradients for up projection
        var upWeightGrads = new float[_inputDim, _hiddenDim];
        var upBiasGrads = new float[_hiddenDim];
        var inputGrads = new float[seqLength, _inputDim];

        for (int inp = 0; inp < _inputDim; inp++)
        {
            for (int hidden = 0; hidden < _hiddenDim; hidden++)
            {
                float weightGrad = 0f;
                for (int seq = 0; seq < seqLength; seq++)
                {
                    weightGrad += _lastInput[seq, inp] * upOutputGrads[seq, hidden];
                    inputGrads[seq, inp] += _upProjection[inp, hidden] * upOutputGrads[seq, hidden];
                }
                upWeightGrads[inp, hidden] = weightGrad;
            }
        }

        // Bias gradients for up projection
        for (int hidden = 0; hidden < _hiddenDim; hidden++)
        {
            for (int seq = 0; seq < seqLength; seq++)
            {
                upBiasGrads[hidden] += upOutputGrads[seq, hidden];
            }
        }

        // Combine all weight gradients
        var allWeightGrads = CombineFFWeightGradients(upWeightGrads, downWeightGrads, upBiasGrads, downBiasGrads);

        return new LayerGradients(
            WeightGradients: allWeightGrads,
            InputGradients: inputGrads,
            BiasGradients: CombineBiasGradients(upBiasGrads, downBiasGrads)
        );
    }

    /// <summary>
    /// Update weights using computed gradients
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        // Split gradients and update each component
        var (upWeightGrads, downWeightGrads, upBiasGrads, downBiasGrads) = SplitFFGradients(gradients.WeightGradients);

        optimizer.UpdateWeights($"{Name}_up_weights", _upProjection.Flatten(), upWeightGrads.Flatten());
        optimizer.UpdateWeights($"{Name}_down_weights", _downProjection.Flatten(), downWeightGrads.Flatten());
        optimizer.UpdateWeights($"{Name}_up_bias", _upBias, upBiasGrads);
        optimizer.UpdateWeights($"{Name}_down_bias", _downBias, downBiasGrads);
    }

    /// <summary>
    /// Initialize feed-forward weights
    /// </summary>
    public void InitializeWeights(Random random)
    {
        // Up projection: Xavier initialization
        WeightInitialization.XavierNormal(_upProjection.Flatten(), _inputDim, _hiddenDim, random);

        // Down projection: Xavier initialization  
        WeightInitialization.XavierNormal(_downProjection.Flatten(), _hiddenDim, _inputDim, random);

        // Initialize biases to zero
        Array.Fill(_upBias, 0f);
        Array.Fill(_downBias, 0f);
    }

    public LayerState GetState()
    {
        return new LayerState(
            LayerType: "FeedForward",
            Weights: new Dictionary<string, float[]>
            {
                ["up_projection"] = _upProjection.Flatten(),
                ["down_projection"] = _downProjection.Flatten(),
                ["up_bias"] = _upBias,
                ["down_bias"] = _downBias
            },
            Metadata: new Dictionary<string, object>
            {
                ["input_dim"] = _inputDim,
                ["hidden_dim"] = _hiddenDim,
                ["activation_type"] = _activationType.ToString()
            }
        );
    }

    public void LoadState(LayerState state)
    {
        if (state.LayerType != "FeedForward")
            throw new ArgumentException($"Expected FeedForward layer, got {state.LayerType}");

        _upProjection = state.Weights["up_projection"].Unflatten(_inputDim, _hiddenDim);
        _downProjection = state.Weights["down_projection"].Unflatten(_hiddenDim, _inputDim);
        _upBias = state.Weights["up_bias"];
        _downBias = state.Weights["down_bias"];
    }

    // Helper methods for gradient handling
    private float[,] CombineFFWeightGradients(float[,] upWeights, float[,] downWeights, float[] upBias, float[] downBias)
    {
        // Simplified combination - in practice you'd want a more structured approach
        int totalSize = upWeights.Length + downWeights.Length + upBias.Length + downBias.Length;
        return new float[1, totalSize];
    }

    private float[] CombineBiasGradients(float[] upBias, float[] downBias)
    {
        var combined = new float[upBias.Length + downBias.Length];
        Array.Copy(upBias, 0, combined, 0, upBias.Length);
        Array.Copy(downBias, 0, combined, upBias.Length, downBias.Length);
        return combined;
    }

    private (float[,] upWeights, float[,] downWeights, float[] upBias, float[] downBias) SplitFFGradients(float[,] combined)
    {
        // Simplified - in practice you'd properly split the gradients
        return (
            new float[_inputDim, _hiddenDim],
            new float[_hiddenDim, _inputDim],
            new float[_hiddenDim],
            new float[_inputDim]
        );
    }
}