using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using ManagedCuda;
//using ManagedCuda.BasicLinearAlgebra;
//using ManagedCuda.CudaBlas;

namespace Core.Mathematics;
/// <summary>
/// High-performance matrix operations for neural networks
/// </summary>
public static class MatrixOperations
{
    //private static readonly bool _cudaAvailable;

    //static MatrixOperations()
    //{
    //    try
    //    {
    //        _cudaAvailable = CudaContext.GetDeviceCount() > 0;
    //    }
    //    catch
    //    {
    //        _cudaAvailable = false;
    //    }
    //}

    //public static bool IsCudaAvailable => _cudaAvailable;

    /// <summary>
    /// Matrix multiplication: C = A * B
    /// A: [aRows x aCols], B: [aCols x bCols] -> C: [aRows x bCols]
    /// Automatically uses CUDA when available.
    /// </summary>
    public static void MatrixMultiply(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result,
        int aRows, int aCols, int bCols)
    {
        //if (_cudaAvailable)
        //{
        //    MatrixMultiplyCuda(a, b, result, aRows, aCols, bCols);
        //    return;
        //}

        MatrixMultiplyCpu(a, b, result, aRows, aCols, bCols);
    }

    private static void MatrixMultiplyCpu(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result,
        int aRows, int aCols, int bCols)
    {
        if (a.Length != (aRows * aCols))
            throw new ArgumentException($"Matrix A size mismatch: expected {aRows * aCols}, got {a.Length}");
        if (b.Length != (aCols * bCols))
            throw new ArgumentException($"Matrix B size mismatch: expected {aCols * bCols}, got {b.Length}");
        if (result.Length != (aRows * bCols))
            throw new ArgumentException($"Result matrix size mismatch: expected {aRows * bCols}, got {result.Length}");

        const int blockSize = 64; // Optimize for L1 cache

        for (int ii = 0; ii < aRows; ii += blockSize)
        {
            for (int jj = 0; jj < bCols; jj += blockSize)
            {
                for (int kk = 0; kk < aCols; kk += blockSize)
                {
                    int iMax = System.Math.Min(ii + blockSize, aRows);
                    int jMax = System.Math.Min(jj + blockSize, bCols);
                    int kMax = System.Math.Min(kk + blockSize, aCols);

                    for (int i = ii; i < iMax; i++)
                    {
                        for (int j = jj; j < jMax; j++)
                        {
                            float sum = 0f;
                            for (int k = kk; k < kMax; k++)
                            {
                                sum += a[(i * aCols) + k] * b[(k * bCols) + j];
                            }
                            if (kk == 0)
                                result[(i * bCols) + j] = sum;
                            else
                                result[(i * bCols) + j] += sum;
                        }
                    }
                }
            }
        }
    }

    //private static void MatrixMultiplyCuda(
    //    ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result,
    //    int aRows, int aCols, int bCols)
    //{
    //    using var context = new CudaContext();
    //    using var cublas = new CudaBlas();

    //    var devA = new CudaDeviceVariable<float>(aRows * aCols);
    //    var devB = new CudaDeviceVariable<float>(aCols * bCols);
    //    var devC = new CudaDeviceVariable<float>(aRows * bCols);

    //    devA.CopyToDevice(a.ToArray());
    //    devB.CopyToDevice(b.ToArray());

    //    const float alpha = 1.0f;
    //    const float beta = 0.0f;
    //    cublas.Gemm(Operation.NonTranspose, Operation.NonTranspose,
    //        aRows, bCols, aCols,
    //        alpha,
    //        devA.DevicePointer, aRows,
    //        devB.DevicePointer, aCols,
    //        beta,
    //        devC.DevicePointer, aRows);

    //    var hostResult = new float[aRows * bCols];
    //    devC.CopyToHost(hostResult);
    //    hostResult.CopyTo(result);

    //    devA.Dispose();
    //    devB.Dispose();
    //    devC.Dispose();
    //}

    /// <summary>
    /// Matrix-vector multiplication: y = A * x
    /// A: [rows x cols], x: [cols] -> y: [rows]
    /// </summary>
    public static void MatrixVectorMultiply(
        ReadOnlySpan<float> matrix, ReadOnlySpan<float> vector, Span<float> result,
        int rows, int cols)
    {
        if (matrix.Length != rows * cols)
            throw new ArgumentException("Matrix size mismatch");
        if (vector.Length != cols)
            throw new ArgumentException("Vector size mismatch");
        if (result.Length != rows)
            throw new ArgumentException("Result size mismatch");

        for (int i = 0; i < rows; i++)
        {
            float sum = 0f;
            for (int j = 0; j < cols; j++)
            {
                sum += matrix[(i * cols) + j] * vector[j];
            }
            result[i] = sum;
        }
    }

    /// <summary>
    /// Transpose matrix: B = A^T
    /// A: [rows x cols] -> B: [cols x rows]
    /// </summary>
    public static void Transpose(
        ReadOnlySpan<float> input, Span<float> output,
        int rows, int cols)
    {
        if (input.Length != (rows * cols))
            throw new ArgumentException("Input matrix size mismatch");
        if (output.Length != (rows * cols))
            throw new ArgumentException("Output matrix size mismatch");

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                output[(j * rows) + i] = input[(i * cols) + j];
            }
        }
    }

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All arrays must have the same length");

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    /// <summary>
    /// Element-wise multiplication (Hadamard product): C = A ⊙ B
    /// </summary>
    public static void ElementwiseMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All arrays must have the same length");

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] * b[i];
        }
    }

    /// <summary>
    /// Scalar multiplication: B = α * A
    /// </summary>
    public static void ScalarMultiply(ReadOnlySpan<float> input, float scalar, Span<float> result)
    {
        if (input.Length != result.Length)
            throw new ArgumentException("Input and result must have the same length");

        for (int i = 0; i < input.Length; i++)
        {
            result[i] = input[i] * scalar;
        }
    }

    /// <summary>
    /// Copy matrix data
    /// </summary>
    public static void Copy(ReadOnlySpan<float> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length");

        source.CopyTo(destination);
    }

    /// <summary>
    /// Zero out a matrix
    /// </summary>
    public static void Zero(Span<float> matrix)
    {
        matrix.Clear();
    }

    /// <summary>
    /// Fill matrix with a constant value
    /// </summary>
    public static void Fill(Span<float> matrix, float value)
    {
        matrix.Fill(value);
    }
}