using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;
using Core.Mathematics;
using Core.Models;

namespace Core.Optimizers;
/// <summary>
/// Stochastic Gradient Descent optimizer with optional momentum
/// Simple but effective baseline optimizer
/// </summary>
public sealed class SGDOptimizer : IOptimizer
{
    private readonly float _momentum;
    private readonly float _weightDecay;
    private readonly bool _useMomentum;
    private readonly Dictionary<string, float[]> _velocities;
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

    public SGDOptimizer(float learningRate = 0.01f, float momentum = 0.9f, float weightDecay = 0.0f)
    {
        if (learningRate <= 0f)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
        if (momentum < 0f || momentum >= 1f)
            throw new ArgumentException("Momentum must be in [0, 1)", nameof(momentum));
        if (weightDecay < 0f)
            throw new ArgumentException("Weight decay must be non-negative", nameof(weightDecay));

        _learningRate = learningRate;
        _momentum = momentum;
        _weightDecay = weightDecay;
        _useMomentum = momentum > 0f;
        _velocities = new Dictionary<string, float[]>();
        _step = 0;
    }

    /// <summary>
    /// Update weights using SGD with optional momentum
    /// weights = weights - learning_rate * (gradients + weight_decay * weights)
    /// With momentum: velocity = momentum * velocity + gradients, weights = weights - learning_rate * velocity
    /// </summary>
    public void UpdateWeights(string parameterName, Span<float> weights, ReadOnlySpan<float> gradients)
    {
        if (weights.Length != gradients.Length)
            throw new ArgumentException("Weights and gradients must have the same length");

        lock (_lock)
        {
            _step++;

            if (_useMomentum)
            {
                UpdateWithMomentum(parameterName, weights, gradients);
            }
            else
            {
                UpdateWithoutMomentum(weights, gradients);
            }
        }
    }

    private void UpdateWithMomentum(string parameterName, Span<float> weights, ReadOnlySpan<float> gradients)
    {
        // Initialize velocity if first time
        if (!_velocities.TryGetValue(parameterName, out var velocity))
        {
            velocity = new float[weights.Length];
            _velocities[parameterName] = velocity;
        }

        // Update velocity and weights
        for (int i = 0; i < weights.Length; i++)
        {
            // Apply weight decay to gradients
            float grad = gradients[i];
            if (_weightDecay > 0f)
            {
                grad += _weightDecay * weights[i];
            }

            // Update velocity: v = momentum * v + grad
            velocity[i] = (_momentum * velocity[i]) + grad;

            // Update weights: w = w - lr * v
            weights[i] -= _learningRate * velocity[i];
        }
    }

    private void UpdateWithoutMomentum(Span<float> weights, ReadOnlySpan<float> gradients)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            // Apply weight decay to gradients
            float grad = gradients[i];
            if (_weightDecay > 0f)
            {
                grad += _weightDecay * weights[i];
            }

            // Update weights: w = w - lr * grad
            weights[i] -= _learningRate * grad;
        }
    }

    /// <summary>
    /// Reset optimizer state (clear momentum terms)
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _velocities.Clear();
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
            foreach (var (key, value) in _velocities)
            {
                momentsCopy[key] = (float[])value.Clone();
            }

            return new OptimizerState(
                Type: OptimizerTypeEnum.SGD,
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
        if (state.Type != OptimizerTypeEnum.SGD)
            throw new ArgumentException($"Expected SGD optimizer state, got {state.Type}");

        lock (_lock)
        {
            _velocities.Clear();
            foreach (var (key, value) in state.Moments)
            {
                _velocities[key] = (float[])value.Clone();
            }

            _step = state.Step;
            _learningRate = state.LearningRate;
        }
    }

    /// <summary>
    /// Get statistics about the optimizer state (for debugging)
    /// </summary>
    public OptimizerStatistics GetStatistics()
    {
        lock (_lock)
        {
            var stats = new Dictionary<string, ParameterStatistics>();

            foreach (var (paramName, velocity) in _velocities)
            {
                var velocityStats = ComputeArrayStatistics(velocity);
                stats[paramName] = new ParameterStatistics(
                    MomentumMean: velocityStats.Mean,
                    MomentumStd: velocityStats.StandardDeviation,
                    MomentumMax: velocityStats.Max,
                    SecondMomentMean: 0f, // Not applicable for SGD
                    SecondMomentStd: 0f,
                    SecondMomentMax: 0f
                );
            }

            return new OptimizerStatistics(
                OptimizerType: OptimizerTypeEnum.SGD,
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