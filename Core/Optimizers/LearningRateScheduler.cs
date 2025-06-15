using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core.Abstractions;

namespace Core.Optimizers;
/// <summary>
/// Learning rate scheduler that adjusts the learning rate during training
/// Essential for achieving good convergence in transformer training
/// </summary>
public sealed class LearningRateScheduler
{
    private readonly ScheduleTypeEnum _scheduleType;
    private readonly float _initialLearningRate;
    private readonly float _minLearningRate;
    private readonly int _warmupSteps;
    private readonly float _decayRate;
    private readonly int _decaySteps;

    public LearningRateScheduler(
        ScheduleTypeEnum scheduleType,
        float initialLearningRate,
        float minLearningRate = 0f,
        int warmupSteps = 0,
        float decayRate = 0.1f,
        int decaySteps = 1000)
    {
        if (initialLearningRate <= 0f)
            throw new ArgumentException("Initial learning rate must be positive", nameof(initialLearningRate));
        if (minLearningRate < 0f)
            throw new ArgumentException("Minimum learning rate must be non-negative", nameof(minLearningRate));
        if (warmupSteps < 0)
            throw new ArgumentException("Warmup steps must be non-negative", nameof(warmupSteps));

        _scheduleType = scheduleType;
        _initialLearningRate = initialLearningRate;
        _minLearningRate = minLearningRate;
        _warmupSteps = warmupSteps;
        _decayRate = decayRate;
        _decaySteps = decaySteps;
    }

    /// <summary>
    /// Get the learning rate for a specific training step
    /// </summary>
    /// <param name="step">Current training step (1-based)</param>
    /// <param name="maxSteps">Total number of training steps (for cosine schedule)</param>
    /// <returns>Learning rate for this step</returns>
    public float GetLearningRate(int step, int? maxSteps = null)
    {
        if (step <= 0)
            throw new ArgumentException("Step must be positive", nameof(step));

        // Apply warmup if configured
        if (step <= _warmupSteps)
        {
            return ApplyWarmup(step);
        }

        // Apply the main schedule
        float baseLearningRate = _scheduleType switch
        {
            ScheduleTypeEnum.Constant => _initialLearningRate,
            ScheduleTypeEnum.Linear => GetLinearDecayRate(step, maxSteps),
            ScheduleTypeEnum.Cosine => GetCosineDecayRate(step, maxSteps),
            ScheduleTypeEnum.Exponential => GetExponentialDecayRate(step),
            ScheduleTypeEnum.StepDecay => GetStepDecayRate(step),
            _ => throw new ArgumentException($"Unknown schedule type: {_scheduleType}")
        };

        // Ensure we don't go below minimum learning rate
        return Math.Max(baseLearningRate, _minLearningRate);
    }

    private float ApplyWarmup(int step)
    {
        // Linear warmup from 0 to initial learning rate
        return _initialLearningRate * (step / (float)_warmupSteps);
    }

    private float GetLinearDecayRate(int step, int? maxSteps)
    {
        if (!maxSteps.HasValue)
            throw new ArgumentException("Linear schedule requires maxSteps parameter");

        int effectiveStep = step - _warmupSteps;
        int effectiveMaxSteps = maxSteps.Value - _warmupSteps;

        if (effectiveStep >= effectiveMaxSteps)
            return _minLearningRate;

        float decayFactor = 1f - ((float)effectiveStep / effectiveMaxSteps);
        return _initialLearningRate * decayFactor;
    }

    private float GetCosineDecayRate(int step, int? maxSteps)
    {
        if (!maxSteps.HasValue)
            throw new ArgumentException("Cosine schedule requires maxSteps parameter");

        int effectiveStep = step - _warmupSteps;
        int effectiveMaxSteps = maxSteps.Value - _warmupSteps;

        if (effectiveStep >= effectiveMaxSteps)
            return _minLearningRate;

        // Cosine annealing: lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + cos(π * step / max_steps))
        float cosineDecay = 0.5f * (1f + MathF.Cos(MathF.PI * effectiveStep / effectiveMaxSteps));
        return _minLearningRate + ((_initialLearningRate - _minLearningRate) * cosineDecay);
    }

    private float GetExponentialDecayRate(int step)
    {
        int effectiveStep = step - _warmupSteps;
        return _initialLearningRate * MathF.Pow(_decayRate, effectiveStep / (float)_decaySteps);
    }

    private float GetStepDecayRate(int step)
    {
        int effectiveStep = step - _warmupSteps;
        int numDecays = effectiveStep / _decaySteps;
        return _initialLearningRate * MathF.Pow(_decayRate, numDecays);
    }

    /// <summary>
    /// Create a scheduler from configuration
    /// </summary>
    public static LearningRateScheduler FromConfiguration(OptimizerConfiguration config)
    {
        if (config.Schedule == null)
        {
            return new LearningRateScheduler(ScheduleTypeEnum.Constant, config.LearningRate);
        }

        return new LearningRateScheduler(
            scheduleType: config.Schedule.Type,
            initialLearningRate: config.LearningRate,
            minLearningRate: config.Schedule.MinLearningRate,
            warmupSteps: config.Schedule.WarmupSteps,
            decayRate: config.Schedule.DecayRate,
            decaySteps: config.Schedule.DecaySteps
        );
    }

    /// <summary>
    /// Get a preview of learning rate schedule (for plotting/visualization)
    /// </summary>
    public float[] GetSchedulePreview(int maxSteps, int previewPoints = 100)
    {
        var preview = new float[previewPoints];

        for (int i = 0; i < previewPoints; i++)
        {
            int step = (int)((i + 1) * (maxSteps / (float)previewPoints));
            preview[i] = GetLearningRate(step, maxSteps);
        }

        return preview;
    }
}
