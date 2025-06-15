using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// AdamW optimizer - Adam with decoupled weight decay
/// Often superior to Adam for transformer training
/// </summary>
public sealed class AdamWOptimizer : IOptimizer
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly Dictionary<string, float[]> _firstMoments;
    private readonly Dictionary<string, float[]> _secondMoments;
    private readonly object _lock = new();

    private float _learningRate;
    private int _step;

    public float LearningRate
    {
        get { lock (_lock) return _learningRate; }
        set { lock (_lock) _learningRate = value; }
    }

    public int Step
    {
        get { lock (_lock) return _step; }
    }

    public AdamWOptimizer(
        float learningRate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.01f)
    {
        if (learningRate <= 0f)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
        if (beta1 < 0f || beta1 >= 1f)
            throw new ArgumentException("Beta1 must be in [0, 1)", nameof(beta1));
        if (beta2 < 0f || beta2 >= 1f)
            throw new ArgumentException("Beta2 must be in [0, 1)", nameof(beta2));
        if (epsilon <= 0f)
            throw new ArgumentException("Epsilon must be positive", nameof(epsilon));
        if (weightDecay < 0f)
            throw new ArgumentException("Weight decay must be non-negative", nameof(weightDecay));

        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _firstMoments = new Dictionary<string, float[]>();
        _secondMoments = new Dictionary<string, float[]>();
        _step = 0;
    }

    /// <summary>
    /// Update weights using AdamW algorithm (decoupled weight decay)
    /// </summary>
    public void UpdateWeights(string parameterName, Span<float> weights, ReadOnlySpan<float> gradients)
    {
        if (weights.Length != gradients.Length)
            throw new ArgumentException("Weights and gradients must have the same length");

        lock (_lock)
        {
            _step++;

            // Initialize moments if first time
            if (!_firstMoments.TryGetValue(parameterName, out var firstMoment))
            {
                firstMoment = new float[weights.Length];
                _firstMoments[parameterName] = firstMoment;
            }

            if (!_secondMoments.TryGetValue(parameterName, out var secondMoment))
            {
                secondMoment = new float[weights.Length];
                _secondMoments[parameterName] = secondMoment;
            }

            // Bias correction factors
            float beta1Correction = 1f - MathF.Pow(_beta1, _step);
            float beta2Correction = 1f - MathF.Pow(_beta2, _step);

            // Update parameters
            for (int i = 0; i < weights.Length; i++)
            {
                float grad = gradients[i];

                // Update biased first moment estimate (no weight decay here)
                firstMoment[i] = (_beta1 * firstMoment[i]) + ((1f - _beta1) * grad);

                // Update biased second moment estimate
                secondMoment[i] = (_beta2 * secondMoment[i]) + ((1f - _beta2) * (grad * grad));

                // Compute bias-corrected moments
                float firstMomentCorrected = firstMoment[i] / beta1Correction;
                float secondMomentCorrected = secondMoment[i] / beta2Correction;

                // AdamW: Apply weight decay directly to weights (decoupled)
                // θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
                float adamUpdate = firstMomentCorrected / (MathF.Sqrt(secondMomentCorrected) + _epsilon);
                float weightDecayUpdate = _weightDecay * weights[i];

                weights[i] -= _learningRate * (adamUpdate + weightDecayUpdate);
            }
        }
    }

    /// <summary>
    /// Reset optimizer state
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _firstMoments.Clear();
            _secondMoments.Clear();
            _step = 0;
        }
    }

    /// <summary>
    /// Get current optimizer state for checkpointing
    /// </summary>
    public OptimizerState GetState()
    {
        lock (_lock)
        {
            var momentsCopy = new Dictionary<string, float[]>();

            foreach (var (key, value) in _firstMoments)
            {
                momentsCopy[$"{key}_first"] = (float[])value.Clone();
            }

            foreach (var (key, value) in _secondMoments)
            {
                momentsCopy[$"{key}_second"] = (float[])value.Clone();
            }

            return new OptimizerState(
                Type: OptimizerTypeEnum.AdamW,
                Moments: momentsCopy,
                Step: _step,
                LearningRate: _learningRate
            );
        }
    }

    /// <summary>
    /// Load optimizer state from checkpoint
    /// </summary>
    public void LoadState(OptimizerState state)
    {
        if (state.Type != OptimizerTypeEnum.AdamW)
            throw new ArgumentException($"Expected AdamW optimizer state, got {state.Type}");

        lock (_lock)
        {
            _firstMoments.Clear();
            _secondMoments.Clear();

            foreach (var (key, value) in state.Moments)
            {
                if (key.EndsWith("_first"))
                {
                    var paramName = key.Substring(0, key.Length - 6);
                    _firstMoments[paramName] = (float[])value.Clone();
                }
                else if (key.EndsWith("_second"))
                {
                    var paramName = key.Substring(0, key.Length - 7);
                    _secondMoments[paramName] = (float[])value.Clone();
                }
            }

            _step = state.Step;
            _learningRate = state.LearningRate;
        }
    }

    /// <summary>
    /// Get statistics about AdamW's state
    /// </summary>
    public OptimizerStatistics GetStatistics()
    {
        lock (_lock)
        {
            var stats = new Dictionary<string, ParameterStatistics>();

            foreach (var (paramName, firstMoment) in _firstMoments)
            {
                if (_secondMoments.TryGetValue(paramName, out var secondMoment))
                {
                    var firstStats = ComputeArrayStatistics(firstMoment);
                    var secondStats = ComputeArrayStatistics(secondMoment);

                    stats[paramName] = new ParameterStatistics(
                        MomentumMean: firstStats.Mean,
                        MomentumStd: firstStats.StandardDeviation,
                        MomentumMax: firstStats.Max,
                        SecondMomentMean: secondStats.Mean,
                        SecondMomentStd: secondStats.StandardDeviation,
                        SecondMomentMax: secondStats.Max
                    );
                }
            }

            return new OptimizerStatistics(
                OptimizerType: OptimizerTypeEnum.AdamW,
                Step: _step,
                LearningRate: _learningRate,
                ParameterStats: stats
            );
        }
    }

    private static ArrayStatistics ComputeArrayStatistics(float[] array)
    {
        if (array.Length == 0)
            return new ArrayStatistics(0f, 0f, 0f, 0f);

        float sum = 0f;
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (float value in array)
        {
            sum += value;
            if (value < min) min = value;
            if (value > max) max = value;
        }

        float mean = sum / array.Length;

        float sumSquaredDiff = 0f;
        foreach (float value in array)
        {
            float diff = value - mean;
            sumSquaredDiff += (diff * diff);
        }

        float standardDeviation = MathF.Sqrt(sumSquaredDiff / array.Length);

        return new ArrayStatistics(mean, standardDeviation, min, max);
    }
}
