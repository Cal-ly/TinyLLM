using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Models;
/// <summary>
/// Adds positional information to token embeddings using sinusoidal encoding
/// </summary>
public sealed class PositionalEmbedding
{
    private readonly float[,] _positionalEncodings; // [max_length, embedding_dim]
    private readonly int _maxLength;
    private readonly int _embeddingDim;

    public PositionalEmbedding(int maxLength, int embeddingDim)
    {
        _maxLength = maxLength;
        _embeddingDim = embeddingDim;
        _positionalEncodings = new float[maxLength, embeddingDim];

        GeneratePositionalEncodings();
    }

    /// <summary>
    /// Add positional encodings to embeddings in-place
    /// </summary>
    /// <param name="embeddings">Token embeddings [sequence_length, embedding_dim]</param>
    /// <param name="sequenceLength">Actual sequence length</param>
    public void AddPositionalEmbeddings(float[,] embeddings, int sequenceLength)
    {
        if (sequenceLength > _maxLength)
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxLength}");

        int embeddingDim = embeddings.GetLength(1);
        if (embeddingDim != _embeddingDim)
            throw new ArgumentException($"Embedding dimension mismatch: expected {_embeddingDim}, got {embeddingDim}");

        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int dim = 0; dim < embeddingDim; dim++)
            {
                embeddings[pos, dim] += _positionalEncodings[pos, dim];
            }
        }
    }

    /// <summary>
    /// Generate sinusoidal positional encodings
    /// </summary>
    private void GeneratePositionalEncodings()
    {
        for (int pos = 0; pos < _maxLength; pos++)
        {
            for (int i = 0; i < _embeddingDim; i++)
            {
                float angle = pos / MathF.Pow(10000f, (2f * i) / _embeddingDim);

                if (i % 2 == 0)
                    _positionalEncodings[pos, i] = MathF.Sin(angle);
                else
                    _positionalEncodings[pos, i] = MathF.Cos(angle);
            }
        }
    }
}