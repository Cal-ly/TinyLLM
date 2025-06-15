using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Mathematics;
/// <summary>
/// Weight initialization strategies for neural networks
/// </summary>
public static class WeightInitialization
{
    /// <summary>
    /// Xavier/Glorot uniform initialization
    /// Draws from uniform distribution [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
    /// </summary>
    public static void XavierUniform(Span<float> weights, int fanIn, int fanOut, Random random)
    {
        float limit = MathF.Sqrt(6f / (fanIn + fanOut));
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = ((float)((random.NextDouble() * 2.0) - 1.0)) * limit;
        }
    }

    /// <summary>
    /// Xavier/Glorot normal initialization
    /// Draws from normal distribution with std = sqrt(2 / (fan_in + fan_out))
    /// </summary>
    public static void XavierNormal(Span<float> weights, int fanIn, int fanOut, Random random)
    {
        float stddev = MathF.Sqrt(2f / (fanIn + fanOut));
        NumericalFunctions.RandomNormal(weights, random, mean: 0f, stddev: stddev);
    }

    /// <summary>
    /// Kaiming/He uniform initialization
    /// Draws from uniform distribution [-limit, limit] where limit = sqrt(6 / fan_in)
    /// </summary>
    public static void KaimingUniform(Span<float> weights, int fanIn, Random random)
    {
        float limit = MathF.Sqrt(6f / fanIn);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = ((float)((random.NextDouble() * 2.0) - 1.0)) * limit;
        }
    }

    /// <summary>
    /// Kaiming/He normal initialization
    /// Draws from normal distribution with std = sqrt(2 / fan_in)
    /// </summary>
    public static void KaimingNormal(Span<float> weights, int fanIn, Random random)
    {
        float stddev = MathF.Sqrt(2f / fanIn);
        NumericalFunctions.RandomNormal(weights, random, mean: 0f, stddev: stddev);
    }

    /// <summary>
    /// Initialize weights based on the specified strategy
    /// </summary>
    public static void Initialize(WeightInitializationEnum initType, Span<float> weights, int fanIn, int fanOut, Random random)
    {
        switch (initType)
        {
            case WeightInitializationEnum.Xavier:
                XavierNormal(weights, fanIn, fanOut, random);
                break;
            case WeightInitializationEnum.Kaiming:
                KaimingNormal(weights, fanIn, random);
                break;
            case WeightInitializationEnum.Normal:
                NumericalFunctions.RandomNormal(weights, random, mean: 0f, stddev: 0.02f);
                break;
            case WeightInitializationEnum.Uniform:
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] = ((float)((random.NextDouble() * 2.0) - 1.0)) * 0.1f;
                }
                break;
            default:
                throw new ArgumentException($"Unknown weight initialization type: {initType}");
        }
    }
}