using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// Gradient clipping utilities to prevent exploding gradients
/// Essential for stable transformer training
/// </summary>
public static class GradientClipper
{
    /// <summary>
    /// Clip gradients by global norm
    /// </summary>
    /// <param name="gradients">Gradient collection to clip</param>
    /// <param name="maxNorm">Maximum allowed gradient norm</param>
    /// <returns>The actual norm before clipping (for monitoring)</returns>
    public static float ClipByGlobalNorm(GradientCollection gradients, float maxNorm)
    {
        if (maxNorm <= 0f)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        // Compute global norm across all parameters
        float globalNormSquared = 0f;
        var allGradients = new List<float[]>();

        foreach (var paramName in gradients.ParameterNames)
        {
            var grad = gradients.GetGradients(paramName).ToArray();
            allGradients.Add(grad);

            foreach (float g in grad)
            {
                globalNormSquared += (g * g);
            }
        }

        float globalNorm = MathF.Sqrt(globalNormSquared);

        // Clip if necessary
        if (globalNorm > maxNorm)
        {
            float clipRatio = maxNorm / globalNorm;

            // Apply clipping to all gradients
            int paramIndex = 0;
            foreach (var paramName in gradients.ParameterNames)
            {
                var originalGrad = allGradients[paramIndex];
                var clippedGrad = new float[originalGrad.Length];

                for (int i = 0; i < originalGrad.Length; i++)
                {
                    clippedGrad[i] = originalGrad[i] * clipRatio;
                }

                gradients.Add(paramName, clippedGrad); // Replace with clipped version
                paramIndex++;
            }
        }

        return globalNorm;
    }

    /// <summary>
    /// Clip gradients by individual parameter norm
    /// </summary>
    /// <param name="gradients">Gradient collection to clip</param>
    /// <param name="maxNorm">Maximum allowed norm per parameter</param>
    /// <returns>Dictionary of norms before clipping for each parameter</returns>
    public static Dictionary<string, float> ClipByParameterNorm(GradientCollection gradients, float maxNorm)
    {
        if (maxNorm <= 0f)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        var norms = new Dictionary<string, float>();

        foreach (var paramName in gradients.ParameterNames)
        {
            var grad = gradients.GetGradients(paramName).ToArray();

            // Compute parameter norm
            float normSquared = 0f;
            foreach (float g in grad)
            {
                normSquared += (g * g);
            }
            float norm = MathF.Sqrt(normSquared);
            norms[paramName] = norm;

            // Clip if necessary
            if (norm > maxNorm)
            {
                float clipRatio = maxNorm / norm;
                var clippedGrad = new float[grad.Length];

                for (int i = 0; i < grad.Length; i++)
                {
                    clippedGrad[i] = grad[i] * clipRatio;
                }

                gradients.Add(paramName, clippedGrad);
            }
        }

        return norms;
    }

    /// <summary>
    /// Clip gradients by value (element-wise)
    /// </summary>
    /// <param name="gradients">Gradient collection to clip</param>
    /// <param name="minValue">Minimum gradient value</param>
    /// <param name="maxValue">Maximum gradient value</param>
    public static void ClipByValue(GradientCollection gradients, float minValue, float maxValue)
    {
        if (minValue >= maxValue)
            throw new ArgumentException("Min value must be less than max value");

        foreach (var paramName in gradients.ParameterNames)
        {
            var grad = gradients.GetGradients(paramName).ToArray();

            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = Math.Clamp(grad[i], minValue, maxValue);
            }

            gradients.Add(paramName, grad);
        }
    }

    /// <summary>
    /// Compute global gradient norm without clipping (for monitoring)
    /// </summary>
    public static float ComputeGlobalNorm(GradientCollection gradients)
    {
        float globalNormSquared = 0f;

        foreach (var paramName in gradients.ParameterNames)
        {
            var grad = gradients.GetGradients(paramName);

            foreach (float g in grad)
            {
                globalNormSquared += (g * g);
            }
        }

        return MathF.Sqrt(globalNormSquared);
    }
}