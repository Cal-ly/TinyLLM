using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Models;
/// <summary>
/// A single transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
/// This follows the pre-norm architecture which is more stable for training
/// </summary>
public sealed class TransformerBlock : ILayer
{
    private readonly AttentionLayer _attention;
    private readonly FeedForwardLayer _feedForward;
    private readonly LayerNormalization _attentionLayerNorm;
    private readonly LayerNormalization _feedForwardLayerNorm;
    private readonly bool _useLayerNorm;
    private readonly bool _useResidualConnections;
    private readonly int _embeddingDim;

    // Cache for backward pass
    private float[,] _lastInput = new float[0, 0];
    private float[,] _lastAttentionInput = new float[0, 0];
    private float[,] _lastAttentionOutput = new float[0, 0];
    private float[,] _lastFFInput = new float[0, 0];
    private float[,] _lastFFOutput = new float[0, 0];

    public string Name => "TransformerBlock";
    public int ParameterCount => _attention.ParameterCount + _feedForward.ParameterCount +
                               (_useLayerNorm ? _attentionLayerNorm.ParameterCount + _feedForwardLayerNorm.ParameterCount : 0);

    public TransformerBlock(
        int embeddingDim,
        int numAttentionHeads,
        int feedForwardDim,
        float dropoutRate = 0.1f,
        ActivationTypeEnum activationType = ActivationTypeEnum.GELU,
        bool useLayerNorm = true,
        bool useResidualConnections = true)
    {
        _embeddingDim = embeddingDim;
        _useLayerNorm = useLayerNorm;
        _useResidualConnections = useResidualConnections;

        // Initialize sub-layers
        _attention = new AttentionLayer(embeddingDim, numAttentionHeads, dropoutRate);
        _feedForward = new FeedForwardLayer(embeddingDim, feedForwardDim, activationType, dropoutRate);

        if (_useLayerNorm)
        {
            _attentionLayerNorm = new LayerNormalization(embeddingDim);
            _feedForwardLayerNorm = new LayerNormalization(embeddingDim);
        }
        else
        {
            _attentionLayerNorm = null!;
            _feedForwardLayerNorm = null!;
        }
    }

    /// <summary>
    /// Forward pass through transformer block
    /// Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    /// </summary>
    public float[,] Forward(float[,] input)
    {
        _lastInput = (float[,])input.Clone();

        // 1. Attention sub-layer with pre-normalization
        float[,] attentionInput;
        if (_useLayerNorm)
        {
            attentionInput = _attentionLayerNorm.Forward(input);
        }
        else
        {
            attentionInput = input;
        }
        _lastAttentionInput = attentionInput;

        var attentionOutput = _attention.Forward(attentionInput);
        _lastAttentionOutput = attentionOutput;

        // Add residual connection
        float[,] afterAttention;
        if (_useResidualConnections)
        {
            afterAttention = AddResidual(input, attentionOutput);
        }
        else
        {
            afterAttention = attentionOutput;
        }

        // 2. Feed-forward sub-layer with pre-normalization
        float[,] ffInput;
        if (_useLayerNorm)
        {
            ffInput = _feedForwardLayerNorm.Forward(afterAttention);
        }
        else
        {
            ffInput = afterAttention;
        }
        _lastFFInput = ffInput;

        var ffOutput = _feedForward.Forward(ffInput);
        _lastFFOutput = ffOutput;

        // Add residual connection
        float[,] output;
        if (_useResidualConnections)
        {
            output = AddResidual(afterAttention, ffOutput);
        }
        else
        {
            output = ffOutput;
        }

        return output;
    }

    /// <summary>
    /// Backward pass through transformer block, returning gradients for each sub-component
    /// </summary>
    public Dictionary<string, float[]> BackwardDetailed(float[,] outputGradients)
    {
        var gradients = new Dictionary<string, float[]>();

        // Current gradient flowing backward
        var currentGrad = outputGradients;

        // Backward through second residual connection
        float[,] ffGradient, afterAttentionGrad;
        if (_useResidualConnections)
        {
            // Both branches get the same gradient for residual connections
            ffGradient = currentGrad;
            afterAttentionGrad = (float[,])currentGrad.Clone();
        }
        else
        {
            ffGradient = currentGrad;
            afterAttentionGrad = new float[currentGrad.GetLength(0), currentGrad.GetLength(1)];
        }

        // Backward through feed-forward layer
        var ffLayerGrads = _feedForward.Backward(ffGradient);
        gradients["ff_weights"] = ffLayerGrads.WeightGradients.Flatten();
        gradients["ff_bias"] = ffLayerGrads.BiasGradients ?? Array.Empty<float>();

        // Backward through feed-forward layer norm
        LayerGradients ffNormGrads;
        if (_useLayerNorm)
        {
            ffNormGrads = _feedForwardLayerNorm.Backward(ffLayerGrads.InputGradients);
            gradients["ff_layer_norm_scale"] = ffNormGrads.WeightGradients.Flatten();
            gradients["ff_layer_norm_bias"] = ffNormGrads.BiasGradients ?? Array.Empty<float>();

            // Add gradients from both paths (residual)
            afterAttentionGrad = AddGradients(afterAttentionGrad, ffNormGrads.InputGradients);
        }
        else
        {
            afterAttentionGrad = AddGradients(afterAttentionGrad, ffLayerGrads.InputGradients);
        }

        // Backward through first residual connection
        float[,] attentionGradient, inputGrad;
        if (_useResidualConnections)
        {
            attentionGradient = afterAttentionGrad;
            inputGrad = (float[,])afterAttentionGrad.Clone();
        }
        else
        {
            attentionGradient = afterAttentionGrad;
            inputGrad = new float[afterAttentionGrad.GetLength(0), afterAttentionGrad.GetLength(1)];
        }

        // Backward through attention layer
        var attentionLayerGrads = _attention.Backward(attentionGradient);
        gradients["attention_weights"] = attentionLayerGrads.WeightGradients.Flatten();

        // Backward through attention layer norm
        if (_useLayerNorm)
        {
            var attentionNormGrads = _attentionLayerNorm.Backward(attentionLayerGrads.InputGradients);
            gradients["attention_layer_norm_scale"] = attentionNormGrads.WeightGradients.Flatten();
            gradients["attention_layer_norm_bias"] = attentionNormGrads.BiasGradients ?? Array.Empty<float>();

            inputGrad = AddGradients(inputGrad, attentionNormGrads.InputGradients);
        }
        else
        {
            inputGrad = AddGradients(inputGrad, attentionLayerGrads.InputGradients);
        }

        // Add "input_gradients" for the parent layer to use
        gradients["input_gradients"] = inputGrad.Flatten();

        return gradients;
    }

    LayerGradients ILayer.Backward(float[,] outputGradients)
    {
        var grads = BackwardDetailed(outputGradients);
        var inputGrad = grads["input_gradients"].Unflatten(outputGradients.GetLength(0), _embeddingDim);
        return new LayerGradients(
            WeightGradients: CombineAllGradients(grads),
            InputGradients: inputGrad
        );
    }

    /// <summary>
    /// Apply gradients to all sub-layers
    /// </summary>
    public void ApplyGradients(Dictionary<string, float[]> layerGradients, IOptimizer optimizer)
    {
        // Apply gradients to attention layer
        if (layerGradients.TryGetValue("attention_weights", out var attentionGrads))
        {
            var attentionLayerGrads = new LayerGradients(
                WeightGradients: attentionGrads.Unflatten(1, attentionGrads.Length),
                InputGradients: new float[0, 0]
            );
            _attention.UpdateWeights(attentionLayerGrads, optimizer);
        }

        // Apply gradients to attention layer norm
        if (_useLayerNorm &&
            layerGradients.TryGetValue("attention_layer_norm_scale", out var attentionNormScale) &&
            layerGradients.TryGetValue("attention_layer_norm_bias", out var attentionNormBias))
        {
            _attentionLayerNorm.ApplyGradients(attentionNormScale, attentionNormBias, optimizer);
        }

        // Apply gradients to feed-forward layer
        if (layerGradients.TryGetValue("ff_weights", out var ffGrads))
        {
            var ffLayerGrads = new LayerGradients(
                WeightGradients: ffGrads.Unflatten(1, ffGrads.Length),
                InputGradients: new float[0, 0],
                BiasGradients: layerGradients.GetValueOrDefault("ff_bias")
            );
            _feedForward.UpdateWeights(ffLayerGrads, optimizer);
        }

        // Apply gradients to feed-forward layer norm
        if (_useLayerNorm &&
            layerGradients.TryGetValue("ff_layer_norm_scale", out var ffNormScale) &&
            layerGradients.TryGetValue("ff_layer_norm_bias", out var ffNormBias))
        {
            _feedForwardLayerNorm.ApplyGradients(ffNormScale, ffNormBias, optimizer);
        }
    }

    /// <summary>
    /// Update weights using computed gradients
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        // This is called by the main model with extracted gradients
        throw new NotSupportedException("Use ApplyGradients method for transformer blocks");
    }

    /// <summary>
    /// Initialize all weights in the transformer block
    /// </summary>
    public void InitializeWeights(Random random, WeightInitializationEnum initType = WeightInitializationEnum.Xavier)
    {
        _attention.InitializeWeights(random);
        _feedForward.InitializeWeights(random);

        if (_useLayerNorm)
        {
            _attentionLayerNorm.InitializeWeights(random);
            _feedForwardLayerNorm.InitializeWeights(random);
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
    /// Get all parameters from this block
    /// </summary>
    public Dictionary<string, float[]> GetParameters()
    {
        var parameters = new Dictionary<string, float[]>();

        // Attention parameters
        var attentionState = _attention.GetState();
        foreach (var (name, weights) in attentionState.Weights)
        {
            parameters[$"attention_{name}"] = weights;
        }

        // Feed-forward parameters
        var ffState = _feedForward.GetState();
        foreach (var (name, weights) in ffState.Weights)
        {
            parameters[$"ff_{name}"] = weights;
        }

        // Layer norm parameters
        if (_useLayerNorm)
        {
            var (attentionScale, attentionBias) = _attentionLayerNorm.GetParameters();
            parameters["attention_layer_norm_scale"] = attentionScale;
            parameters["attention_layer_norm_bias"] = attentionBias;

            var (ffScale, ffBias) = _feedForwardLayerNorm.GetParameters();
            parameters["ff_layer_norm_scale"] = ffScale;
            parameters["ff_layer_norm_bias"] = ffBias;
        }

        return parameters;
    }

    /// <summary>
    /// Load parameters into this block
    /// </summary>
    public void LoadParameters(Dictionary<string, float[]> parameters)
    {
        // Load attention parameters
        var attentionParams = ExtractPrefixedParams(parameters, "attention_", excludePrefixes: new[] { "attention_layer_norm_" });
        var attentionState = new LayerState("MultiHeadAttention", attentionParams, new Dictionary<string, object>());
        _attention.LoadState(attentionState);

        // Load feed-forward parameters
        var ffParams = ExtractPrefixedParams(parameters, "ff_", excludePrefixes: new[] { "ff_layer_norm_" });
        var ffState = new LayerState("FeedForward", ffParams, new Dictionary<string, object>());
        _feedForward.LoadState(ffState);

        // Load layer norm parameters
        if (_useLayerNorm)
        {
            if (parameters.TryGetValue("attention_layer_norm_scale", out var attentionScale) &&
                parameters.TryGetValue("attention_layer_norm_bias", out var attentionBias))
            {
                _attentionLayerNorm.LoadParameters(attentionScale, attentionBias);
            }

            if (parameters.TryGetValue("ff_layer_norm_scale", out var ffScale) &&
                parameters.TryGetValue("ff_layer_norm_bias", out var ffBias))
            {
                _feedForwardLayerNorm.LoadParameters(ffScale, ffBias);
            }
        }
    }

    public LayerState GetState()
    {
        return new LayerState(
            LayerType: "TransformerBlock",
            Weights: GetParameters(),
            Metadata: new Dictionary<string, object>
            {
                ["embedding_dim"] = _embeddingDim,
                ["use_layer_norm"] = _useLayerNorm,
                ["use_residual_connections"] = _useResidualConnections
            }
        );
    }

    public void LoadState(LayerState state)
    {
        if (state.LayerType != "TransformerBlock")
            throw new ArgumentException($"Expected TransformerBlock layer, got {state.LayerType}");

        LoadParameters(state.Weights);
    }

    // Helper methods
    private static float[,] AddResidual(float[,] input, float[,] output)
    {
        int seqLength = input.GetLength(0);
        int embeddingDim = input.GetLength(1);
        var result = new float[seqLength, embeddingDim];

        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < embeddingDim; j++)
            {
                result[i, j] = input[i, j] + output[i, j];
            }
        }

        return result;
    }

    private static float[,] AddGradients(float[,] grad1, float[,] grad2)
    {
        int rows = grad1.GetLength(0);
        int cols = grad1.GetLength(1);
        var result = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = grad1[i, j] + grad2[i, j];
            }
        }

        return result;
    }

    private static float[,] CombineAllGradients(Dictionary<string, float[]> gradients)
    {
        var totalSize = gradients.Values.Sum(g => g.Length);
        return new float[1, totalSize]; // Simplified
    }

    private static Dictionary<string, float[]> ExtractPrefixedParams(Dictionary<string, float[]> parameters, string prefix, string[] excludePrefixes)
    {
        var extracted = new Dictionary<string, float[]>();

        foreach (var (name, values) in parameters)
        {
            if (name.StartsWith(prefix) && !excludePrefixes.Any(ex => name.StartsWith(ex)))
            {
                var shortName = name.Substring(prefix.Length);
                extracted[shortName] = values;
            }
        }

        return extracted;
    }
}