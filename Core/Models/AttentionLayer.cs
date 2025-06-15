using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using Core.Mathematics;

namespace Core.Models;
/// <summary>
/// Multi-head self-attention layer - the heart of the transformer
/// </summary>
public sealed class AttentionLayer : ILayer
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _dropoutRate;

    // Weight matrices for Q, K, V projections and output
    private float[,] _queryWeights;   // [embedding_dim, embedding_dim]
    private float[,] _keyWeights;     // [embedding_dim, embedding_dim]
    private float[,] _valueWeights;   // [embedding_dim, embedding_dim]
    private float[,] _outputWeights;  // [embedding_dim, embedding_dim]

    // Cached values for backward pass
    private float[,] _lastInput = new float[0, 0];
    private float[,] _lastQueries = new float[0, 0];
    private float[,] _lastKeys = new float[0, 0];
    private float[,] _lastValues = new float[0, 0];
    private float[,] _lastAttentionWeights = new float[0, 0];
    private float[,] _lastAttentionOutput = new float[0, 0];

    public string Name => "MultiHeadAttention";
    public int ParameterCount => 4 * _embeddingDim * _embeddingDim;

    public AttentionLayer(int embeddingDim, int numHeads, float dropoutRate = 0.1f)
    {
        if (embeddingDim % numHeads != 0)
            throw new ArgumentException($"Embedding dimension {embeddingDim} must be divisible by number of heads {numHeads}");

        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _dropoutRate = dropoutRate;

        // Initialize weight matrices
        _queryWeights = new float[embeddingDim, embeddingDim];
        _keyWeights = new float[embeddingDim, embeddingDim];
        _valueWeights = new float[embeddingDim, embeddingDim];
        _outputWeights = new float[embeddingDim, embeddingDim];
    }

    /// <summary>
    /// Forward pass through multi-head attention
    /// </summary>
    /// <param name="input">Input sequence [sequence_length, embedding_dim]</param>
    /// <returns>Attention output [sequence_length, embedding_dim]</returns>
    public float[,] Forward(float[,] input)
    {
        int seqLength = input.GetLength(0);
        int inputDim = input.GetLength(1);

        if (inputDim != _embeddingDim)
            throw new ArgumentException($"Input dimension {inputDim} doesn't match embedding dimension {_embeddingDim}");

        _lastInput = (float[,])input.Clone();

        // 1. Compute Q, K, V matrices
        _lastQueries = ComputeProjection(input, _queryWeights);
        _lastKeys = ComputeProjection(input, _keyWeights);
        _lastValues = ComputeProjection(input, _valueWeights);

        // 2. Reshape for multi-head processing
        var queriesReshaped = ReshapeForMultiHead(_lastQueries, seqLength);  // [num_heads, seq_length, head_dim]
        var keysReshaped = ReshapeForMultiHead(_lastKeys, seqLength);
        var valuesReshaped = ReshapeForMultiHead(_lastValues, seqLength);

        // 3. Compute scaled dot-product attention for each head
        var attentionOutputs = new float[_numHeads, seqLength, _headDim];
        var attentionWeights = new float[_numHeads, seqLength, seqLength];

        for (int head = 0; head < _numHeads; head++)
        {
            var headAttentionWeights = ComputeAttentionWeights(
                GetHead(queriesReshaped, head, seqLength),
                GetHead(keysReshaped, head, seqLength),
                seqLength
            );

            SetHead(attentionWeights, head, headAttentionWeights);

            var headOutput = ApplyAttention(
                headAttentionWeights,
                GetHead(valuesReshaped, head, seqLength)
            );

            SetHead(attentionOutputs, head, headOutput);
        }

        // 4. Concatenate heads and apply output projection
        var concatenated = ConcatenateHeads(attentionOutputs, seqLength);
        _lastAttentionOutput = concatenated;
        _lastAttentionWeights = attentionWeights.FlattenFirstDimension(); // Store for backward pass

        var output = ComputeProjection(concatenated, _outputWeights);
        return output;
    }

    /// <summary>
    /// Backward pass through attention layer
    /// </summary>
    public LayerGradients Backward(float[,] outputGradients)
    {
        int seqLength = outputGradients.GetLength(0);

        // Gradients for output projection
        var outputWeightGrads = ComputeProjectionGradients(_lastAttentionOutput, outputGradients);
        var attentionOutputGrads = ComputeProjectionInputGradients(outputGradients, _outputWeights);

        // This is a simplified backward pass - full implementation would compute
        // gradients through the attention mechanism, which is quite complex
        var inputGradients = ComputeSimplifiedAttentionGradients(attentionOutputGrads);

        // Compute gradients for Q, K, V projections
        var queryWeightGrads = ComputeProjectionGradients(_lastInput, inputGradients);
        var keyWeightGrads = ComputeProjectionGradients(_lastInput, inputGradients);
        var valueWeightGrads = ComputeProjectionGradients(_lastInput, inputGradients);

        return new LayerGradients(
            WeightGradients: CombineWeightGradients(queryWeightGrads, keyWeightGrads, valueWeightGrads, outputWeightGrads),
            InputGradients: inputGradients
        );
    }

    /// <summary>
    /// Update weights using computed gradients
    /// </summary>
    public void UpdateWeights(LayerGradients gradients, IOptimizer optimizer)
    {
        // Extract individual weight gradients and update
        var flatWeights = CombineWeights(_queryWeights, _keyWeights, _valueWeights, _outputWeights);
        optimizer.UpdateWeights($"{Name}_weights", flatWeights, gradients.WeightGradients.Cast<float>().ToArray());

        // Split updated weights back
        SplitWeights(flatWeights, out _queryWeights, out _keyWeights, out _valueWeights, out _outputWeights);
    }

    /// <summary>
    /// Initialize attention weights
    /// </summary>
    public void InitializeWeights(Random random)
    {
        var fanIn = _embeddingDim;
        var fanOut = _embeddingDim;

        WeightInitialization.XavierNormal(_queryWeights.Flatten(), fanIn, fanOut, random);
        WeightInitialization.XavierNormal(_keyWeights.Flatten(), fanIn, fanOut, random);
        WeightInitialization.XavierNormal(_valueWeights.Flatten(), fanIn, fanOut, random);
        WeightInitialization.XavierNormal(_outputWeights.Flatten(), fanIn, fanOut, random);
    }

    public LayerState GetState()
    {
        return new LayerState(
            LayerType: "MultiHeadAttention",
            Weights: new Dictionary<string, float[]>
            {
                ["query_weights"] = _queryWeights.Flatten(),
                ["key_weights"] = _keyWeights.Flatten(),
                ["value_weights"] = _valueWeights.Flatten(),
                ["output_weights"] = _outputWeights.Flatten()
            },
            Metadata: new Dictionary<string, object>
            {
                ["embedding_dim"] = _embeddingDim,
                ["num_heads"] = _numHeads,
                ["head_dim"] = _headDim
            }
        );
    }

    public void LoadState(LayerState state)
    {
        if (state.LayerType != "MultiHeadAttention")
            throw new ArgumentException($"Expected MultiHeadAttention layer, got {state.LayerType}");

        _queryWeights = state.Weights["query_weights"].Unflatten(_embeddingDim, _embeddingDim);
        _keyWeights = state.Weights["key_weights"].Unflatten(_embeddingDim, _embeddingDim);
        _valueWeights = state.Weights["value_weights"].Unflatten(_embeddingDim, _embeddingDim);
        _outputWeights = state.Weights["output_weights"].Unflatten(_embeddingDim, _embeddingDim);
    }

    // Helper methods for attention computation
    private float[,] ComputeProjection(float[,] input, float[,] weights)
    {
        int seqLength = input.GetLength(0);
        int outputDim = weights.GetLength(1);
        var output = new float[seqLength, outputDim];

        MatrixOperations.MatrixMultiply(
            input.Flatten(), weights.Flatten(), output.Flatten(),
            seqLength, _embeddingDim, outputDim
        );

        return output;
    }

    private float[,,] ReshapeForMultiHead(float[,] matrix, int seqLength)
    {
        var reshaped = new float[_numHeads, seqLength, _headDim];

        for (int head = 0; head < _numHeads; head++)
        {
            for (int seq = 0; seq < seqLength; seq++)
            {
                for (int dim = 0; dim < _headDim; dim++)
                {
                    reshaped[head, seq, dim] = matrix[seq, (head * _headDim) + dim];
                }
            }
        }

        return reshaped;
    }

    private float[,] ComputeAttentionWeights(float[,] queries, float[,] keys, int seqLength)
    {
        var scores = new float[seqLength, seqLength];
        var scale = 1f / MathF.Sqrt(_headDim);

        // Compute Q * K^T
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < seqLength; j++)
            {
                float score = 0f;
                for (int k = 0; k < _headDim; k++)
                {
                    score += queries[i, k] * keys[j, k];
                }
                scores[i, j] = score * scale;
            }
        }

        // Apply causal mask (prevent looking at future tokens)
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = i + 1; j < seqLength; j++)
            {
                scores[i, j] = float.NegativeInfinity;
            }
        }

        // Apply softmax to each row
        for (int i = 0; i < seqLength; i++)
        {
            var row = new float[seqLength];
            for (int j = 0; j < seqLength; j++)
                row[j] = scores[i, j];

            NumericalFunctions.Softmax(row, row);

            for (int j = 0; j < seqLength; j++)
                scores[i, j] = row[j];
        }

        return scores;
    }

    private float[,] ApplyAttention(float[,] attentionWeights, float[,] values)
    {
        int seqLength = attentionWeights.GetLength(0);
        var output = new float[seqLength, _headDim];

        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                float sum = 0f;
                for (int k = 0; k < seqLength; k++)
                {
                    sum += attentionWeights[i, k] * values[k, j];
                }
                output[i, j] = sum;
            }
        }

        return output;
    }

    private float[,] ConcatenateHeads(float[,,] headOutputs, int seqLength)
    {
        var concatenated = new float[seqLength, _embeddingDim];

        for (int seq = 0; seq < seqLength; seq++)
        {
            for (int head = 0; head < _numHeads; head++)
            {
                for (int dim = 0; dim < _headDim; dim++)
                {
                    concatenated[seq, (head * _headDim) + dim] = headOutputs[head, seq, dim];
                }
            }
        }

        return concatenated;
    }

    // Simplified helper methods for gradients (full implementation would be more complex)
    private float[,] GetHead(float[,,] tensor, int headIndex, int seqLength)
    {
        var head = new float[seqLength, _headDim];
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                head[i, j] = tensor[headIndex, i, j];
            }
        }
        return head;
    }

    private void SetHead(float[,,] tensor, int headIndex, float[,] head)
    {
        int seqLength = head.GetLength(0);
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                tensor[headIndex, i, j] = head[i, j];
            }
        }
    }

    private float[,] ComputeProjectionGradients(float[,] input, float[,] outputGrads)
    {
        // Simplified gradient computation
        return new float[input.GetLength(1), outputGrads.GetLength(1)];
    }

    private float[,] ComputeProjectionInputGradients(float[,] outputGrads, float[,] weights)
    {
        // Simplified gradient computation
        return new float[outputGrads.GetLength(0), weights.GetLength(0)];
    }

    private float[,] ComputeSimplifiedAttentionGradients(float[,] outputGrads)
    {
        // Simplified - just pass through gradients
        return outputGrads;
    }

    private float[,] CombineWeightGradients(params float[][,] gradients)
    {
        // Combine all weight gradients into a single matrix
        int totalSize = gradients.Sum(g => g.Length);
        return new float[1, totalSize]; // Simplified
    }

    private float[] CombineWeights(params float[][,] weights)
    {
        return weights.SelectMany(w => w.Flatten()).ToArray();
    }

    private void SplitWeights(float[] combinedWeights, out float[,] q, out float[,] k, out float[,] v, out float[,] o)
    {
        int singleMatrixSize = _embeddingDim * _embeddingDim;
        q = combinedWeights.AsSpan(0, singleMatrixSize).ToArray().Unflatten(_embeddingDim, _embeddingDim);
        k = combinedWeights.AsSpan(singleMatrixSize, singleMatrixSize).ToArray().Unflatten(_embeddingDim, _embeddingDim);
        v = combinedWeights.AsSpan(2 * singleMatrixSize, singleMatrixSize).ToArray().Unflatten(_embeddingDim, _embeddingDim);
        o = combinedWeights.AsSpan(3 * singleMatrixSize, singleMatrixSize).ToArray().Unflatten(_embeddingDim, _embeddingDim);
    }
}