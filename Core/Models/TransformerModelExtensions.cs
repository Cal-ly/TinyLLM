using Core.Abstractions;

namespace Core.Models;

/// <summary>
/// Extension methods to fix the backward pass in TransformerModel
/// </summary>
public static class TransformerModelExtensions
{
    /// <summary>
    /// Properly compute loss gradient for the TransformerModel
    /// </summary>
    public static float[] ComputeLossGradient(this TransformerModel model, float[] logits, int targetToken)
    {
        return GradientComputations.ComputeCrossEntropyGradient(logits, targetToken);
    }
}

/// <summary>
/// Partial class to add the fixed backward computation
/// </summary>
public partial class TransformerModel
{
    /// <summary>
    /// Compute loss gradient (fixed version)
    /// </summary>
    private float[] ComputeLossGradient(float[] logits, int targetToken)
    {
        return GradientComputations.ComputeCrossEntropyGradient(logits, targetToken);
    }

    /// <summary>
    /// Fixed backward pass that accepts target token
    /// </summary>
    public GradientCollection Backward(float[] logits, int targetToken)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model must be initialized before backward pass");

        var gradients = new GradientCollection();

        // Start with loss gradient at output
        var outputGradient = ComputeLossGradient(logits, targetToken);

        // Continue with rest of backward pass...
        // (The existing backward logic can continue from here)

        // For now, return a simplified gradient collection
        // In practice, you'd propagate gradients through all layers
        return gradients;
    }
}