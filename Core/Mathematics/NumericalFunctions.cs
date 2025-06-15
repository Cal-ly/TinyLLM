using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Mathematics;

/// <summary>
/// Essential numerical functions for neural networks
/// </summary>
public static class NumericalFunctions
{
    /// <summary>
    /// Softmax function: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
    /// Numerically stable implementation
    /// </summary>
    public static void Softmax(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        if (input.Length == 0)
            return;

        // Find max for numerical stability
        float max = float.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] > max)
                max = input[i];
        }

        // Compute exp(x - max) and sum
        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = MathF.Exp(input[i] - max);
            sum += output[i];
        }

        // Normalize
        if (sum > 0f)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] /= sum;
            }
        }
    }

    /// <summary>
    /// Log-softmax function: log_softmax(x_i) = x_i - log(Σ(exp(x_j)))
    /// Numerically stable implementation
    /// </summary>
    public static void LogSoftmax(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        if (input.Length == 0)
            return;

        // Find max for numerical stability
        float max = float.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] > max)
                max = input[i];
        }

        // Compute sum of exp(x - max)
        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            sum += MathF.Exp(input[i] - max);
        }

        float logSumExp = max + MathF.Log(sum);

        // Compute log_softmax
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = input[i] - logSumExp;
        }
    }

    /// <summary>
    /// Layer normalization: (x - mean) / sqrt(variance + epsilon)
    /// </summary>
    public static void LayerNorm(ReadOnlySpan<float> input, Span<float> output, float epsilon = 1e-6f)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        if (input.Length == 0)
            return;

        // Compute mean
        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }
        float mean = sum / input.Length;

        // Compute variance
        float sumSquaredDiff = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            float diff = input[i] - mean;
            sumSquaredDiff += (diff * diff);
        }
        float variance = sumSquaredDiff / input.Length;

        // Normalize
        float invStd = 1f / MathF.Sqrt(variance + epsilon);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = (input[i] - mean) * invStd;
        }
    }

    /// <summary>
    /// Layer normalization with learnable scale and bias parameters
    /// </summary>
    public static void LayerNorm(ReadOnlySpan<float> input, Span<float> output,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, float epsilon = 1e-6f)
    {
        if (input.Length != output.Length || input.Length != scale.Length || input.Length != bias.Length)
            throw new ArgumentException("All arrays must have the same length");

        // First compute normalized values
        LayerNorm(input, output, epsilon);

        // Apply scale and bias
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = (output[i] * scale[i]) + bias[i];
        }
    }

    /// <summary>
    /// Compute L2 (Euclidean) norm of a vector
    /// </summary>
    public static float L2Norm(ReadOnlySpan<float> vector)
    {
        float sumSquares = 0f;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares += (vector[i] * vector[i]);
        }
        return MathF.Sqrt(sumSquares);
    }

    /// <summary>
    /// Clip values to a specified range
    /// </summary>
    public static void Clip(ReadOnlySpan<float> input, Span<float> output, float minValue, float maxValue)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Math.Clamp(input[i], minValue, maxValue);
        }
    }

    /// <summary>
    /// Generate random values from normal distribution (Box-Muller transform)
    /// </summary>
    public static void RandomNormal(Span<float> output, Random random, float mean = 0f, float stddev = 1f)
    {
        for (int i = 0; i < output.Length; i += 2)
        {
            // Box-Muller transform
            float u1 = (float)random.NextDouble();
            float u2 = (float)random.NextDouble();

            float z0 = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            float z1 = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2);

            output[i] = (z0 * stddev) + mean;
            if (i + 1 < output.Length)
                output[i + 1] = (z1 * stddev) + mean;
        }
    }
}