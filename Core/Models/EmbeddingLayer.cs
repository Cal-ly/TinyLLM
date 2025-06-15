using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using Core.Mathematics;

namespace Core.Models;
/// <summary>
/// Token embedding layer that converts token IDs to dense vectors
/// </summary>
public sealed class EmbeddingLayer : ILayer
{
    private float[,] _embeddings; // [vocab_size, embedding_dim]
    private readonly int _vocabSize;
    private readonly int _embeddingDim;

    public string Name => "TokenEmbedding";
    public int ParameterCount => _vocabSize * _embeddingDim;

    public EmbeddingLayer(int vocabSize, int embeddingDim)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        _embeddings = new float[vocabSize, embeddingDim];
    }

    /// <summary>
    /// Forward pass: Convert token IDs to embeddings
    /// </summary>
    /// <param name="tokenIds">Token IDs [sequence_length]</param>
    /// <returns>Embeddings [sequence_length, embedding_dim]</returns>
    public float[,] Forward(ReadOnlySpan<int> tokenIds)
    {
        int seqLength = tokenIds.Length;
        var output = new float[seqLength, _embeddingDim];

        for (int i = 0; i < seqLength; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId < 0 || tokenId >= _vocabSize)
                throw new ArgumentException($"Token ID {tokenId} is out of vocabulary range [0, {_vocabSize - 1}]");

            // Copy embedding for this token
            for (int j = 0; j < _embeddingDim; j++)
            {
                output[i, j] = _embeddings[tokenId, j];
            }
        }

        return output;
    }

    /// <summary>
    /// Overload for ILayer interface (not used for embeddings)
    /// </summary>
    public float[,] Forward(float[,] input)
    {
        throw new NotSupportedException("EmbeddingLayer.Forward should be called with token IDs, not float matrix");
    }

    /// <summary>
    /// Backward pass: Compute gradients for embeddings
    /// </summary>
    /// <param name="tokenIds">Original token IDs</param>
    /// <param name="outputGradients">Gradients from next layer [sequence_length, embedding_dim]</param>
    /// <returns>Gradients for embedding weights</returns>
    public float[] Backward(int[] tokenIds, float[,] outputGradients)
    {
        var embeddingGradients = new float[_vocabSize * _embeddingDim];

        for (int i = 0; i < tokenIds.Length; i++)
        {
            int tokenId = tokenIds[i];
            int baseIndex = tokenId * _embeddingDim;

            for (int j = 0; j < _embeddingDim; j++)
            {
                embeddingGradients[baseIndex + j] += outputGradients[i, j];
            }
        }

        return embeddingGradients;
    }

    /// <summary>
    /// ILayer interface backward pass (not used)
    /// </summary>
    public LayerGradients Backward(float[,] outputGradients)
    {
        throw new NotSupportedException("EmbeddingLayer.Backward should be called with token IDs");
    }

    /// <summary>
    /// Apply gradients to embedding weights
    /// </summary>
    public void ApplyGradients(ReadOnlySpan<float> gradients, IOptimizer optimizer)
    {
        var flatWeights = _embeddings.Flatten();
        optimizer.UpdateWeights("token_embeddings", flatWeights, gradients);
        _embeddings = flatWeights.Unflatten(_vocabSize, _embeddingDim);
    }

    /// <summary>
    /// Update weights using gradients and optimizer
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        throw new NotSupportedException("Use ApplyGradients method for embedding layer");
    }

    /// <summary>
    /// Initialize embedding weights
    /// </summary>
    public void InitializeWeights(Random random, WeightInitializationEnum initType = WeightInitializationEnum.Xavier)
    {
        var flatWeights = _embeddings.Flatten();
        WeightInitialization.Initialize(initType, flatWeights, _vocabSize, _embeddingDim, random);
        _embeddings = flatWeights.Unflatten(_vocabSize, _embeddingDim);
    }

    /// <summary>
    /// ILayer interface initialization
    /// </summary>
    public void InitializeWeights(Random random)
    {
        InitializeWeights(random, WeightInitializationEnum.Xavier);
    }

    /// <summary>
    /// Get current embedding weights
    /// </summary>
    public float[,] GetWeights() => (float[,])_embeddings.Clone();

    /// <summary>
    /// Load embedding weights
    /// </summary>
    public void LoadWeights(float[,] weights)
    {
        if (weights.GetLength(0) != _vocabSize || weights.GetLength(1) != _embeddingDim)
            throw new ArgumentException($"Weight dimensions mismatch: expected [{_vocabSize}, {_embeddingDim}], got [{weights.GetLength(0)}, {weights.GetLength(1)}]");

        _embeddings = (float[,])weights.Clone();
    }

    /// <summary>
    /// Get layer state for serialization
    /// </summary>
    public LayerState GetState()
    {
        return new LayerState(
            LayerType: "TokenEmbedding",
            Weights: new Dictionary<string, float[]> { ["embeddings"] = _embeddings.Flatten() },
            Metadata: new Dictionary<string, object>
            {
                ["vocab_size"] = _vocabSize,
                ["embedding_dim"] = _embeddingDim
            }
        );
    }

    /// <summary>
    /// Load layer state from serialization
    /// </summary>
    public void LoadState(LayerState state)
    {
        if (state.LayerType != "TokenEmbedding")
            throw new ArgumentException($"Expected TokenEmbedding layer, got {state.LayerType}");

        if (state.Weights.TryGetValue("embeddings", out var embeddingWeights))
        {
            _embeddings = embeddingWeights.Unflatten(_vocabSize, _embeddingDim);
        }
    }
}
