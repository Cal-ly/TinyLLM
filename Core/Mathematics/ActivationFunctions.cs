using Core.Abstractions;

namespace Core.Mathematics;
/// <summary>
/// Activation functions and their derivatives for neural networks
/// </summary>
public static class ActivationFunctions
{
    /// <summary>
    /// Rectified Linear Unit: ReLU(x) = max(0, x)
    /// </summary>
    public static void ReLU(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = MathF.Max(0f, input[i]);
        }
    }

    /// <summary>
    /// ReLU derivative: 1 if x > 0, else 0
    /// </summary>
    public static void ReLUDerivative(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = input[i] > 0f ? 1f : 0f;
        }
    }

    /// <summary>
    /// Gaussian Error Linear Unit: GELU(x) = x * Φ(x)
    /// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// </summary>
    public static void GELU(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        const float sqrt2OverPi = 0.7978845608f; // √(2/π)
        const float coeff = 0.044715f;

        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            float x3 = x * x * x;
            float inner = sqrt2OverPi * (x + (coeff * x3));
            output[i] = 0.5f * x * (1f + MathF.Tanh(inner));
        }
    }

    /// <summary>
    /// GELU derivative (approximation)
    /// </summary>
    public static void GELUDerivative(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;

        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            float x2 = x * x;
            float x3 = x2 * x;

            float inner = sqrt2OverPi * (x + (coeff * x3));
            float tanh_inner = MathF.Tanh(inner);
            float sech2_inner = 1f - (tanh_inner * tanh_inner); // sech²(x) = 1 - tanh²(x)

            float gelu_part = 0.5f * (1f + tanh_inner);
            float derivative_part = 0.5f * x * sqrt2OverPi * (1f + (3f * coeff * x2)) * sech2_inner;

            output[i] = gelu_part + derivative_part;
        }
    }

    /// <summary>
    /// Hyperbolic tangent: tanh(x)
    /// </summary>
    public static void Tanh(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = MathF.Tanh(input[i]);
        }
    }

    /// <summary>
    /// Tanh derivative: 1 - tanh²(x)
    /// </summary>
    public static void TanhDerivative(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            float tanh_x = MathF.Tanh(input[i]);
            output[i] = 1f - (tanh_x * tanh_x);
        }
    }

    /// <summary>
    /// Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
    /// </summary>
    public static void Sigmoid(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = 1f / (1f + MathF.Exp(-input[i]));
        }
    }

    /// <summary>
    /// Sigmoid derivative: σ(x) * (1 - σ(x))
    /// </summary>
    public static void SigmoidDerivative(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            float sigmoid_x = 1f / (1f + MathF.Exp(-input[i]));
            output[i] = sigmoid_x * (1f - sigmoid_x);
        }
    }

    /// <summary>
    /// Apply activation function based on type
    /// </summary>
    public static void Apply(ActivationTypeEnum activationType, ReadOnlySpan<float> input, Span<float> output)
    {
        switch (activationType)
        {
            case ActivationTypeEnum.ReLU:
                ReLU(input, output);
                break;
            case ActivationTypeEnum.GELU:
                GELU(input, output);
                break;
            case ActivationTypeEnum.Tanh:
                Tanh(input, output);
                break;
            case ActivationTypeEnum.Sigmoid:
                Sigmoid(input, output);
                break;
            default:
                throw new ArgumentException($"Unknown activation type: {activationType}");
        }
    }

    /// <summary>
    /// Apply activation derivative based on type
    /// </summary>
    public static void ApplyDerivative(ActivationTypeEnum activationType, ReadOnlySpan<float> input, Span<float> output)
    {
        switch (activationType)
        {
            case ActivationTypeEnum.ReLU:
                ReLUDerivative(input, output);
                break;
            case ActivationTypeEnum.GELU:
                GELUDerivative(input, output);
                break;
            case ActivationTypeEnum.Tanh:
                TanhDerivative(input, output);
                break;
            case ActivationTypeEnum.Sigmoid:
                SigmoidDerivative(input, output);
                break;
            default:
                throw new ArgumentException($"Unknown activation type: {activationType}");
        }
    }
}
