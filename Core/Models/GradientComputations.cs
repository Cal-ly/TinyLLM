using Core.Mathematics;

namespace Core.Models;

/// <summary>
/// Fixed gradient computation methods for the TransformerModel
/// </summary>
public static class GradientComputations
{
    /// <summary>
    /// Compute gradient for cross-entropy loss with respect to logits
    /// For softmax + cross-entropy, the gradient is simply: predictions - one_hot(target)
    /// </summary>
    public static float[] ComputeCrossEntropyGradient(float[] logits, int targetToken)
    {
        var probabilities = new float[logits.Length];
        NumericalFunctions.Softmax(logits, probabilities);

        // Gradient is probabilities minus one-hot encoding of target
        var gradients = (float[])probabilities.Clone();
        gradients[targetToken] -= 1.0f;

        return gradients;
    }

    /// <summary>
    /// Compute gradients through matrix multiplication: dL/dA = dL/dC * B^T
    /// </summary>
    public static float[,] MatrixMultiplyGradientLeft(float[,] gradOutput, float[,] rightInput)
    {
        int m = gradOutput.GetLength(0);
        int n = gradOutput.GetLength(1);
        int k = rightInput.GetLength(0);

        var gradInput = new float[m, k];

        // gradInput = gradOutput * rightInput^T
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                float sum = 0f;
                for (int l = 0; l < n; l++)
                {
                    sum += gradOutput[i, l] * rightInput[j, l];
                }
                gradInput[i, j] = sum;
            }
        }

        return gradInput;
    }

    /// <summary>
    /// Compute gradients through matrix multiplication: dL/dB = A^T * dL/dC
    /// </summary>
    public static float[,] MatrixMultiplyGradientRight(float[,] leftInput, float[,] gradOutput)
    {
        int k = leftInput.GetLength(1);
        int n = gradOutput.GetLength(1);
        int m = leftInput.GetLength(0);

        var gradInput = new float[k, n];

        // gradInput = leftInput^T * gradOutput
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int l = 0; l < m; l++)
                {
                    sum += leftInput[l, i] * gradOutput[l, j];
                }
                gradInput[i, j] = sum;
            }
        }

        return gradInput;
    }
}