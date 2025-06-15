using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Mathematics;

/// <summary>
/// Loss functions for neural network training
/// </summary>
public static class LossFunctions
{
    /// <summary>
    /// Cross-entropy loss for multi-class classification
    /// Loss = -Σ(y_true * log(y_pred))
    /// </summary>
    public static float CrossEntropy(ReadOnlySpan<float> predictions, ReadOnlySpan<float> targets)
    {
        if (predictions.Length != targets.Length)
            throw new ArgumentException("Predictions and targets must have the same length");

        float loss = 0f;
        for (int i = 0; i < predictions.Length; i++)
        {
            // Clip predictions to avoid log(0)
            float pred = Math.Clamp(predictions[i], 1e-7f, 1f - 1e-7f);
            loss -= targets[i] * MathF.Log(pred);
        }
        return loss;
    }

    /// <summary>
    /// Sparse cross-entropy loss (target is class index, not one-hot)
    /// </summary>
    public static float SparseCrossEntropy(ReadOnlySpan<float> predictions, int targetClass)
    {
        if (targetClass < 0 || targetClass >= predictions.Length)
            throw new ArgumentException("Target class index out of range");

        // Clip prediction to avoid log(0)
        float pred = Math.Clamp(predictions[targetClass], 1e-7f, 1f - 1e-7f);
        return -MathF.Log(pred);
    }

    /// <summary>
    /// Gradient of cross-entropy loss with respect to predictions
    /// Assumes predictions are softmax outputs
    /// Gradient = y_pred - y_true
    /// </summary>
    public static void CrossEntropyGradient(ReadOnlySpan<float> predictions, ReadOnlySpan<float> targets, Span<float> gradients)
    {
        if (predictions.Length != targets.Length || predictions.Length != gradients.Length)
            throw new ArgumentException("All arrays must have the same length");

        for (int i = 0; i < predictions.Length; i++)
        {
            gradients[i] = predictions[i] - targets[i];
        }
    }

    /// <summary>
    /// Gradient of sparse cross-entropy loss
    /// </summary>
    public static void SparseCrossEntropyGradient(ReadOnlySpan<float> predictions, int targetClass, Span<float> gradients)
    {
        if (targetClass < 0 || targetClass >= predictions.Length)
            throw new ArgumentException("Target class index out of range");
        if (predictions.Length != gradients.Length)
            throw new ArgumentException("Predictions and gradients must have the same length");

        // Copy predictions to gradients
        predictions.CopyTo(gradients);

        // Subtract 1 from the target class
        gradients[targetClass]--;
    }

    /// <summary>
    /// Mean squared error loss
    /// </summary>
    public static float MeanSquaredError(ReadOnlySpan<float> predictions, ReadOnlySpan<float> targets)
    {
        if (predictions.Length != targets.Length)
            throw new ArgumentException("Predictions and targets must have the same length");

        float sumSquaredError = 0f;
        for (int i = 0; i < predictions.Length; i++)
        {
            float diff = predictions[i] - targets[i];
            sumSquaredError += (diff * diff);
        }
        return sumSquaredError / predictions.Length;
    }

    /// <summary>
    /// Gradient of mean squared error loss
    /// </summary>
    public static void MeanSquaredErrorGradient(ReadOnlySpan<float> predictions, ReadOnlySpan<float> targets, Span<float> gradients)
    {
        if (predictions.Length != targets.Length || predictions.Length != gradients.Length)
            throw new ArgumentException("All arrays must have the same length");

        float scale = 2f / predictions.Length;
        for (int i = 0; i < predictions.Length; i++)
        {
            gradients[i] = scale * (predictions[i] - targets[i]);
        }
    }
}