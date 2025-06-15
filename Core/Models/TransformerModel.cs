using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using Core.Abstractions;
using Core.Mathematics;

namespace Core.Models;
/// <summary>
/// GPT-style transformer language model implementation
/// This is the main class that orchestrates all the layers
/// </summary>
public sealed partial class TransformerModel : ILanguageModel
{
    private ModelConfiguration? _config;
    private EmbeddingLayer? _tokenEmbedding;
    private PositionalEmbedding? _positionalEmbedding;
    private TransformerBlock[]? _transformerBlocks;
    private LayerNormalization? _finalLayerNorm;
    private OutputLayer? _outputLayer;

    // State for backward pass
    private int[] _lastInputTokens = Array.Empty<int>();
    private float[,] _lastEmbeddings = new float[0, 0];
    private float[][,] _lastBlockOutputs = new float[0][,];
    private float[,] _lastNormalized = new float[0, 0];
    private float[] _lastLogits = Array.Empty<float>();

    private bool _isInitialized = false;

    public ModelConfiguration Configuration => _config!;
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Initialize the model with the specified configuration
    /// </summary>
    public void Initialize(ModelConfiguration config)
    {
        config.Validate();
        _config = config;

        // Initialize all layers
        _tokenEmbedding = new EmbeddingLayer(_config!.VocabularySize, _config!.EmbeddingDim);
        _positionalEmbedding = new PositionalEmbedding(_config!.ContextLength, _config!.EmbeddingDim);

        // Create transformer blocks
        _transformerBlocks = new TransformerBlock[_config!.NumLayers];
        for (int i = 0; i < _config!.NumLayers; i++)
        {
            _transformerBlocks[i] = new TransformerBlock(
                embeddingDim: _config!.EmbeddingDim,
                numAttentionHeads: _config!.NumAttentionHeads,
                feedForwardDim: _config!.FeedForwardDim,
                dropoutRate: _config!.DropoutRate,
                activationType: _config!.ActivationFunction,
                useLayerNorm: _config!.UseLayerNorm,
                useResidualConnections: _config!.UseResidualConnections
            );
        }

        _finalLayerNorm = new LayerNormalization(_config!.EmbeddingDim);
        _outputLayer = new OutputLayer(_config!.EmbeddingDim, _config!.VocabularySize);

        // Initialize weights
        InitializeWeights();

        _isInitialized = true;
    }

    /// <summary>
    /// Forward pass through the model
    /// </summary>
    /// <param name="inputTokens">Input token sequence [sequence_length]</param>
    /// <returns>Logits for next token prediction [vocabulary_size]</returns>
    public float[] Forward(ReadOnlySpan<int> inputTokens)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before forward pass");

        if (inputTokens.Length == 0)
            throw new ArgumentException("Input tokens cannot be empty");

        if (inputTokens.Length > _config!.ContextLength)
            throw new ArgumentException($"Input sequence length ({inputTokens.Length}) exceeds context length ({_config!.ContextLength})");

        // Store input for backward pass
        _lastInputTokens = inputTokens.ToArray();
        int seqLength = inputTokens.Length;

        // 1. Token embeddings: [seq_length] -> [seq_length, embedding_dim]
        var embeddings = _tokenEmbedding!.Forward(inputTokens);

        // 2. Add positional embeddings
        _positionalEmbedding!.AddPositionalEmbeddings(embeddings, seqLength);
        _lastEmbeddings = (float[,])embeddings.Clone();

        // 3. Pass through transformer blocks
        var hiddenStates = embeddings;
        _lastBlockOutputs = new float[_config!.NumLayers][,];

        for (int i = 0; i < _config!.NumLayers; i++)
        {
            hiddenStates = _transformerBlocks![i].Forward(hiddenStates);
            _lastBlockOutputs[i] = (float[,])hiddenStates.Clone();
        }

        // 4. Final layer normalization
        if (_config!.UseLayerNorm)
        {
            hiddenStates = _finalLayerNorm!.Forward(hiddenStates);
            _lastNormalized = (float[,])hiddenStates.Clone();
        }

        // 5. Output projection - only use the last token for next token prediction
        var lastTokenHidden = ExtractLastToken(hiddenStates, seqLength);
        var logits = _outputLayer!.Forward(lastTokenHidden);
        _lastLogits = (float[])logits.Clone();

        return logits;
    }

    /// <summary>
    /// Backward pass to compute gradients (legacy/simplified)
    /// </summary>
    public GradientCollection Backward(float loss)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before backward pass");

        var gradients = new GradientCollection();

        // Start with loss gradient at output
        var outputGradient = ComputeLossGradient(loss);

        // Backward through output layer
        var outputLayerGrads = _outputLayer!.Backward(outputGradient);
        gradients.Add("output_weights", outputLayerGrads.WeightGradients.Flatten());
        gradients.Add("output_bias", outputLayerGrads.BiasGradients ?? Array.Empty<float>());

        // Expand gradient to full sequence (we only computed gradient for last token)
        var seqLength = _lastInputTokens.Length;
        var fullSequenceGrad = ExpandLastTokenGradient(outputLayerGrads.InputGradients, seqLength, _config!.EmbeddingDim);

        // Backward through final layer norm
        var currentGrad = fullSequenceGrad;
        if (_config!.UseLayerNorm)
        {
            var layerNormGrads = _finalLayerNorm!.Backward(currentGrad);
            var (scaleGrad, biasGrad) = SplitLayerNormGrads(layerNormGrads.WeightGradients, _config!.EmbeddingDim);
            gradients.Add("final_layer_norm_scale", scaleGrad);
            gradients.Add("final_layer_norm_bias", biasGrad);
            currentGrad = layerNormGrads.InputGradients;
        }

        // Backward through transformer blocks (in reverse order)
        for (int i = _config!.NumLayers - 1; i >= 0; i--)
        {
            var blockGrads = _transformerBlocks![i].BackwardDetailed(currentGrad);

            // Collect gradients with layer-specific names
            var layerPrefix = $"layer_{i}";
            foreach (var (name, grad) in blockGrads)
            {
                if (name == "input_gradients")
                    continue;

                gradients.Add($"{layerPrefix}_{name}", grad);
            }

            currentGrad = blockGrads["input_gradients"].Unflatten(seqLength, _config!.EmbeddingDim);
        }

        // Backward through positional embeddings (just pass through)
        // Positional embeddings are fixed, so no gradients to compute

        // Backward through token embeddings
        var embeddingGrads = _tokenEmbedding!.Backward(_lastInputTokens, currentGrad);
        gradients.Add("token_embeddings", embeddingGrads);

        return gradients;
    }

    /// <summary>
    /// Full backward pass: computes gradients for all model parameters.
    /// </summary>
    /// <param name="logits">Logits from the forward pass</param>
    /// <param name="targetToken">Target token index for loss</param>
    /// <returns>GradientCollection for all parameters</returns>
    public GradientCollection Backward(float[] logits, int targetToken)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before backward pass");

        var gradients = new GradientCollection();

        // 1. Compute loss gradient w.r.t. logits (cross-entropy)
        var outputGradient = GradientComputations.ComputeCrossEntropyGradient(logits, targetToken);

        // 2. Backward through output layer (last token only)
        var outputLayerGrads = _outputLayer!.Backward(outputGradient);
        gradients.Add("output_weights", outputLayerGrads.WeightGradients.Flatten());
        if (outputLayerGrads.BiasGradients != null)
            gradients.Add("output_bias", outputLayerGrads.BiasGradients);

        // 3. Expand gradient to full sequence (only last token position gets nonzero gradient)
        int seqLength = _lastInputTokens.Length;
        int embeddingDim = _config!.EmbeddingDim;
        var fullSequenceGrad = ExpandLastTokenGradient(outputLayerGrads.InputGradients, seqLength, embeddingDim);

        // 4. Backward through final layer normalization (if used)
        var currentGrad = fullSequenceGrad;
        if (_config!.UseLayerNorm)
        {
            var layerNormGrads = _finalLayerNorm!.Backward(currentGrad);
            var (scaleGrad, biasGrad) = SplitLayerNormGrads(layerNormGrads.WeightGradients, embeddingDim);
            gradients.Add("final_layer_norm_scale", scaleGrad);
            gradients.Add("final_layer_norm_bias", biasGrad);
            currentGrad = layerNormGrads.InputGradients;
        }

        // 5. Backward through transformer blocks (in reverse order)
        for (int i = _config!.NumLayers - 1; i >= 0; i--)
        {
            var blockGrads = _transformerBlocks![i].BackwardDetailed(currentGrad);

            // Collect gradients with layer-specific names
            var layerPrefix = $"layer_{i}";
            foreach (var (name, grad) in blockGrads)
            {
                if (name == "input_gradients")
                    continue;
                gradients.Add($"{layerPrefix}_{name}", grad);
            }

            // Prepare gradient for next block
            currentGrad = blockGrads["input_gradients"].Unflatten(seqLength, embeddingDim);
        }

        // 6. No gradients for positional embeddings (fixed)

        // 7. Backward through token embeddings
        var embeddingGrads = _tokenEmbedding!.Backward(_lastInputTokens, currentGrad);
        gradients.Add("token_embeddings", embeddingGrads);

        return gradients;
    }

    /// <summary>
    /// Apply gradients using the specified optimizer
    /// </summary>
    public void ApplyGradients(GradientCollection gradients, IOptimizer optimizer)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before applying gradients");

        // Update token embeddings
        if (gradients.HasGradients("token_embeddings"))
        {
            _tokenEmbedding!.ApplyGradients(gradients.GetGradients("token_embeddings"), optimizer);
        }

        // Update transformer blocks
        for (int i = 0; i < _config!.NumLayers; i++)
        {
            var layerPrefix = $"layer_{i}";
            var layerGradients = ExtractLayerGradients(gradients, layerPrefix);
            _transformerBlocks![i].ApplyGradients(layerGradients, optimizer);
        }

        // Update final layer norm
        if (_config!.UseLayerNorm && gradients.HasGradients("final_layer_norm_scale"))
        {
            _finalLayerNorm!.ApplyGradients(
                gradients.GetGradients("final_layer_norm_scale"),
                gradients.GetGradients("final_layer_norm_bias"),
                optimizer
            );
        }

        // Update output layer
        if (gradients.HasGradients("output_weights"))
        {
            _outputLayer!.ApplyGradients(
                gradients.GetGradients("output_weights"),
                gradients.GetGradients("output_bias"),
                optimizer
            );
        }
    }

    /// <summary>
    /// Get the current state of the model for serialization
    /// </summary>
    public ModelState GetState()
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before getting state");

        var parameters = new Dictionary<string, float[]>();

        // Token embeddings
        parameters["token_embeddings"] = _tokenEmbedding!.GetWeights().Flatten();

        // Transformer blocks
        for (int i = 0; i < _config!.NumLayers; i++)
        {
            var layerParams = _transformerBlocks![i].GetParameters();
            foreach (var (name, weights) in layerParams)
            {
                parameters[$"layer_{i}_{name}"] = weights;
            }
        }

        // Final layer norm
        if (_config!.UseLayerNorm)
        {
            var (scale, bias) = _finalLayerNorm!.GetParameters();
            parameters["final_layer_norm_scale"] = scale;
            parameters["final_layer_norm_bias"] = bias;
        }

        // Output layer
        var (outputWeights, outputBias) = _outputLayer!.GetParameters();
        parameters["output_weights"] = outputWeights.Flatten();
        if (outputBias != null)
            parameters["output_bias"] = outputBias;

        return new ModelState(_config!, parameters);
    }

    /// <summary>
    /// Load model state from serialization
    /// </summary>
    public void LoadState(ModelState state)
    {
        Initialize(state.Configuration);

        // Load token embeddings
        if (state.Parameters.TryGetValue("token_embeddings", out var tokenEmbeddings))
        {
            _tokenEmbedding!.LoadWeights(tokenEmbeddings.Unflatten(_config!.VocabularySize, _config!.EmbeddingDim));
        }

        // Load transformer blocks
        for (int i = 0; i < _config!.NumLayers; i++)
        {
            var layerParams = ExtractLayerParameters(state.Parameters, $"layer_{i}");
            _transformerBlocks![i].LoadParameters(layerParams);
        }

        // Load final layer norm
        if (_config!.UseLayerNorm)
        {
            if (state.Parameters.TryGetValue("final_layer_norm_scale", out var scale) &&
                state.Parameters.TryGetValue("final_layer_norm_bias", out var bias))
            {
                _finalLayerNorm!.LoadParameters(scale, bias);
            }
        }

        // Load output layer
        if (state.Parameters.TryGetValue("output_weights", out var outputWeights))
        {
            var outputBias = state.Parameters.GetValueOrDefault("output_bias");
            var weightsMatrix = outputWeights.Unflatten(_config!.EmbeddingDim, _config!.VocabularySize);
            _outputLayer!.LoadParameters(weightsMatrix, outputBias);
        }
    }

    /// <summary>
    /// Reset the model to initial state
    /// </summary>
    public void Reset()
    {
        _isInitialized = false;
        _lastInputTokens = Array.Empty<int>();
        _lastEmbeddings = new float[0, 0];
        _lastBlockOutputs = new float[_config!.NumLayers][,];
        _lastNormalized = new float[0, 0];
        _lastLogits = Array.Empty<float>();
    }

    /// <summary>
    /// Initialize all model weights using the specified strategy
    /// </summary>
    private void InitializeWeights()
    {
        var random = new Random(42); // Fixed seed for reproducibility

        // Initialize token embeddings
        _tokenEmbedding!.InitializeWeights(random, _config!.WeightInit);

        // Initialize transformer blocks
        foreach (var block in _transformerBlocks!)
        {
            block.InitializeWeights(random, _config!.WeightInit);
        }

        // Initialize final layer norm
        if (_config!.UseLayerNorm)
        {
            _finalLayerNorm!.InitializeWeights(random);
        }

        // Initialize output layer
        _outputLayer!.InitializeWeights(random, _config!.WeightInit);
    }

    /// <summary>
    /// Extract the last token's hidden state for prediction
    /// </summary>
    private static float[] ExtractLastToken(float[,] hiddenStates, int seqLength)
    {
        int embeddingDim = hiddenStates.GetLength(1);
        var lastToken = new float[embeddingDim];

        for (int i = 0; i < embeddingDim; i++)
        {
            lastToken[i] = hiddenStates[seqLength - 1, i];
        }

        return lastToken;
    }

    /// <summary>
    /// Compute loss gradient (simplified for now)
    /// </summary>
    private float[] ComputeLossGradient(float loss)
    {
        // For now, return a simple gradient
        // In practice, this would be computed by the training loop
        var gradient = new float[_config!.VocabularySize];
        gradient[0] = loss; // Simplified
        return gradient;
    }

    /// <summary>
    /// Compute loss gradient (cross-entropy) for logits and target token.
    /// </summary>
    private float[] ComputeLossGradient(float[] logits, int targetToken)
    {
        return GradientComputations.ComputeCrossEntropyGradient(logits, targetToken);
    }

    /// <summary>
    /// Expand gradient from last token to full sequence
    /// </summary>
    private static float[,] ExpandLastTokenGradient(float[,] lastTokenGrad, int seqLength, int embeddingDim)
    {
        var fullGrad = new float[seqLength, embeddingDim];

        // Only the last token position gets gradients
        for (int i = 0; i < embeddingDim; i++)
        {
            fullGrad[seqLength - 1, i] = lastTokenGrad[0, i];
        }

        return fullGrad;
    }

    private static (float[] scale, float[] bias) SplitLayerNormGrads(float[,] combined, int dim)
    {
        var scale = new float[dim];
        var bias = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            scale[i] = combined[0, i];
            bias[i] = combined[0, dim + i];
        }
        return (scale, bias);
    }

    /// <summary>
    /// Extract gradients for a specific layer
    /// </summary>
    private static Dictionary<string, float[]> ExtractLayerGradients(GradientCollection gradients, string layerPrefix)
    {
        var layerGrads = new Dictionary<string, float[]>();

        foreach (var paramName in gradients.ParameterNames)
        {
            if (paramName.StartsWith(layerPrefix + "_"))
            {
                var shortName = paramName.Substring(layerPrefix.Length + 1);
                layerGrads[shortName] = gradients.GetGradients(paramName).ToArray();
            }
        }

        return layerGrads;
    }

    /// <summary>
    /// Extract parameters for a specific layer
    /// </summary>
    private static Dictionary<string, float[]> ExtractLayerParameters(Dictionary<string, float[]> parameters, string layerPrefix)
    {
        var layerParams = new Dictionary<string, float[]>();

        foreach (var (paramName, values) in parameters)
        {
            if (paramName.StartsWith(layerPrefix + "_"))
            {
                var shortName = paramName.Substring(layerPrefix.Length + 1);
                layerParams[shortName] = values;
            }
        }

        return layerParams;
    }
}
