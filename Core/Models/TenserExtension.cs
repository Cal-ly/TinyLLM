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
}