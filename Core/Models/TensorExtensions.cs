using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Models;
public static class TensorExtensions
{
    /// <summary>
    /// Flatten the first dimension of a 3D array
    /// </summary>
    public static float[,] FlattenFirstDimension(this float[,,] array)
    {
        int dim0 = array.GetLength(0);
        int dim1 = array.GetLength(1);
        int dim2 = array.GetLength(2);

        var result = new float[dim1, dim2];

        // Take the first slice for simplicity 
        // TODO: might want to average or sum?
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                result[i, j] = array[0, i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Flatten a 2D matrix into a 1D array in row-major order
    /// </summary>
    public static float[] Flatten(this float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new float[rows * cols];
        int index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[index++] = matrix[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Convert a flattened array back into a 2D matrix
    /// </summary>
    public static float[,] Unflatten(this float[] data, int rows, int cols)
    {
        if (data.Length != rows * cols)
            throw new ArgumentException($"Array length {data.Length} does not match dimensions [{rows}, {cols}]");

        var result = new float[rows, cols];
        int index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = data[index++];
            }
        }
        return result;
    }
}