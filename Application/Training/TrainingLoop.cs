using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Services.Training;
/// <summary>
/// Core training loop that handles epochs, batches, and training steps
/// </summary>
public sealed class TrainingLoop
{
    private readonly ILogger _logger;
    private readonly MetricsLogger _metricsLogger;

    public TrainingLoop(ILogger logger, MetricsLogger metricsLogger)
    {
        _logger = logger;
        _metricsLogger = metricsLogger;
    }

    /// <summary>
    /// Run the complete training loop
    /// </summary>
    public async Task<TrainingResult> RunAsync(
        TrainingContext context,
        int startEpoch = 0,
        CancellationToken cancellationToken = default)
    {
        var config = context.Configuration;
        var model = context.Model;
        var optimizer = context.Optimizer;
        var scheduler = context.Scheduler;
        var lossComputer = context.LossComputer;

        var bestValidationLoss = float.MaxValue;
        var bestModel = model.GetState();
        var totalSteps = 0;
        var stepsWithoutImprovement = 0;

        var trainingHistory = new List<EpochMetrics>();

        for (int epoch = startEpoch; epoch < config.NumEpochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            _logger.LogInformation("Starting epoch {Epoch}/{TotalEpochs}", epoch + 1, config.NumEpochs);
            var epochStartTime = DateTime.UtcNow;

            // Training phase
            var epochMetrics = await RunEpochAsync(
                context, epoch, totalSteps, cancellationToken);

            totalSteps += epochMetrics.TotalSteps;
            trainingHistory.Add(epochMetrics);

            // Validation phase
            ValidationMetrics? validationMetrics = null;
            if (context.ValidationDataset != null && (epoch + 1) % config.ValidationFrequency == 0)
            {
                _logger.LogInformation("Running validation for epoch {Epoch}", epoch + 1);
                var validationConfig = new EvaluationConfiguration
                {
                    BatchSize = config.BatchSize,
                    MaxBatches = config.MaxValidationBatches
                };

                var validationResult = await context.ValidationRunner.EvaluateAsync(
                    model, context.ValidationDataset, validationConfig, cancellationToken);

                validationMetrics = new ValidationMetrics(
                    Loss: validationResult.AverageLoss,
                    Perplexity: validationResult.Perplexity,
                    Accuracy: validationResult.Accuracy
                );

                _logger.LogInformation("Validation - Loss: {Loss:F4}, Perplexity: {Perplexity:F2}",
                    validationMetrics.Loss, validationMetrics.Perplexity);

                // Check for improvement
                if (validationMetrics.Loss < bestValidationLoss)
                {
                    bestValidationLoss = validationMetrics.Loss;
                    bestModel = model.GetState();
                    stepsWithoutImprovement = 0;

                    // Save best model
                    await context.CheckpointManager.SaveCheckpointAsync(
                        model, optimizer, epoch, validationMetrics.Loss,
                        Path.Combine(config.OutputDirectory, "best_model.ckpt"),
                        cancellationToken);
                }
                else
                {
                    stepsWithoutImprovement++;
                }
            }

            // Log epoch summary
            var epochDuration = DateTime.UtcNow - epochStartTime;
            _logger.LogInformation(
                "Epoch {Epoch} completed - Train Loss: {TrainLoss:F4}, LR: {LR:F6}, Time: {Duration}",
                epoch + 1, epochMetrics.AverageLoss, epochMetrics.FinalLearningRate, epochDuration);

            // Log metrics
            await _metricsLogger.LogEpochAsync(epoch, epochMetrics, validationMetrics, cancellationToken);

            // Save regular checkpoint
            if ((epoch + 1) % config.CheckpointFrequency == 0)
            {
                var checkpointPath = Path.Combine(config.OutputDirectory, $"checkpoint_epoch_{epoch + 1}.ckpt");
                await context.CheckpointManager.SaveCheckpointAsync(
                    model, optimizer, epoch, epochMetrics.AverageLoss, checkpointPath, cancellationToken);
            }

            // Early stopping check
            if (config.EarlyStoppingPatience > 0 && stepsWithoutImprovement >= config.EarlyStoppingPatience)
            {
                _logger.LogInformation("Early stopping triggered after {Steps} epochs without improvement", stepsWithoutImprovement);
                break;
            }

            // Generate sample text
            if ((epoch + 1) % config.SampleGenerationFrequency == 0)
            {
                await GenerateSampleTextAsync(context, epoch, cancellationToken);
            }
        }

        // Load best model for final result
        model.LoadState(bestModel);

        return new TrainingResult(
            Success: true,
            TotalEpochs: trainingHistory.Count,
            TotalSteps: totalSteps,
            BestValidationLoss: bestValidationLoss,
            TrainingHistory: trainingHistory,
            FinalModelState: bestModel
        );
    }

    private async Task<EpochMetrics> RunEpochAsync(
        TrainingContext context,
        int epoch,
        int globalStep,
        CancellationToken cancellationToken)
    {
        var config = context.Configuration;
        var model = context.Model;
        var optimizer = context.Optimizer;
        var scheduler = context.Scheduler;
        var lossComputer = context.LossComputer;

        var batches = context.TrainDataset.CreateBatches(
            config.BatchSize,
            config.SequenceLength,
            shuffle: true,
            seed: config.RandomSeed + epoch);

        float totalLoss = 0f;
        int batchCount = 0;
        int currentStep = globalStep;

        foreach (var batch in batches)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            currentStep++;

            // Update learning rate
            optimizer.LearningRate = scheduler.GetLearningRate(currentStep, config.GetTotalSteps());

            // Training step
            var stepResult = await RunTrainingStepAsync(model, optimizer, lossComputer, batch, config, cancellationToken);

            totalLoss += stepResult.Loss;
            batchCount++;

            // Log step metrics
            if (currentStep % config.LoggingFrequency == 0)
            {
                _logger.LogDebug(
                    "Step {Step}: Loss={Loss:F4}, LR={LR:F6}, GradNorm={GradNorm:F3}",
                    currentStep, stepResult.Loss, optimizer.LearningRate, stepResult.GradientNorm);

                await _metricsLogger.LogStepAsync(currentStep, stepResult, cancellationToken);
            }

            // Memory cleanup
            if (batchCount % 100 == 0)
            {
                GC.Collect(0, GCCollectionMode.Optimized);
            }
        }

        var avgLoss = totalLoss / Math.Max(batchCount, 1);
        return new EpochMetrics(
            Epoch: epoch,
            AverageLoss: avgLoss,
            TotalSteps: batchCount,
            FinalLearningRate: optimizer.LearningRate,
            GradientNorm: 0f // Would track this during training
        );
    }

    private async Task<TrainingStepResult> RunTrainingStepAsync(
        ILanguageModel model,
        IOptimizer optimizer,
        LossComputer lossComputer,
        TrainingBatch batch,
        TrainingConfiguration config,
        CancellationToken cancellationToken)
    {
        await Task.Yield(); // Allow for async pattern

        float totalLoss = 0f;
        var allGradients = new GradientCollection();

        // Process each sequence in the batch
        for (int i = 0; i < batch.BatchSize; i++)
        {
            var inputTokens = batch.GetInputSequence(i);
            var targetTokens = batch.GetTargetSequence(i);

            // Forward pass
            var logits = model.Forward(inputTokens);

            // Compute loss for last token prediction
            var targetToken = targetTokens[targetTokens.Length - 1];
            var loss = lossComputer.ComputeLoss(logits, targetToken);
            totalLoss += loss;

            // Backward pass
            var gradients = model.Backward(loss);

            // Accumulate gradients
            AccumulateGradients(allGradients, gradients);
        }

        // Average gradients across batch
        AverageGradients(allGradients, batch.BatchSize);

        // Apply gradient clipping
        var gradientNorm = GradientClipper.ClipByGlobalNorm(allGradients, config.Optimizer.GradientClipNorm);

        // Apply gradients
        model.ApplyGradients(allGradients, optimizer);

        var avgLoss = totalLoss / batch.BatchSize;
        return new TrainingStepResult(avgLoss, gradientNorm);
    }

    private static void AccumulateGradients(GradientCollection target, GradientCollection source)
    {
        foreach (var paramName in source.ParameterNames)
        {
            var sourceGrads = source.GetGradients(paramName).ToArray();

            if (target.HasGradients(paramName))
            {
                var targetGrads = target.GetGradients(paramName).ToArray();
                for (int i = 0; i < sourceGrads.Length; i++)
                {
                    targetGrads[i] += sourceGrads[i];
                }
                target.Add(paramName, targetGrads);
            }
            else
            {
                target.Add(paramName, sourceGrads);
            }
        }
    }

    private static void AverageGradients(GradientCollection gradients, int batchSize)
    {
        foreach (var paramName in gradients.ParameterNames)
        {
            var grads = gradients.GetGradients(paramName).ToArray();
            for (int i = 0; i < grads.Length; i++)
            {
                grads[i] /= batchSize;
            }
            gradients.Add(paramName, grads);
        }
    }

    private async Task GenerateSampleTextAsync(
        TrainingContext context,
        int epoch,
        CancellationToken cancellationToken)
    {
        try
        {
            var samplePrompts = context.Configuration.SamplePrompts;
            if (samplePrompts.Length == 0)
                return;

            _logger.LogInformation("Generating sample text for epoch {Epoch}", epoch + 1);

            foreach (var prompt in samplePrompts.Take(3)) // Limit to 3 samples
            {
                var generated = await GenerateTextSampleAsync(context.Model, context.TrainDataset.Tokenizer, prompt, cancellationToken);
                _logger.LogInformation("Sample: '{Prompt}' -> '{Generated}'", prompt, generated);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to generate sample text");
        }
    }

    private async Task<string> GenerateTextSampleAsync(
        ILanguageModel model,
        ITokenizer tokenizer,
        string prompt,
        CancellationToken cancellationToken)
    {
        await Task.Yield();

        var tokens = tokenizer.Encode(prompt).ToList();
        var random = new Random();

        for (int i = 0; i < 50; i++) // Generate up to 50 tokens
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            var contextTokens = tokens.Skip(Math.Max(0, tokens.Count - model.Configuration.ContextLength)).ToArray();
            var logits = model.Forward(contextTokens);

            // Simple temperature sampling
            var probabilities = new float[logits.Length];
            var scaledLogits = logits.Select(l => l / 0.8f).ToArray(); // Temperature = 0.8
            NumericalFunctions.Softmax(scaledLogits, probabilities);

            var nextToken = SampleFromDistribution(probabilities, random);
            tokens.Add(nextToken);

            // Check for natural stopping
            if (tokenizer is CharacterTokenizer charTokenizer)
            {
                var character = charTokenizer.GetCharacter(nextToken);
                if (character == '.' || character == '!' || character == '?')
                    break;
            }
        }

        return tokenizer.Decode(tokens.ToArray());
    }

    private static int SampleFromDistribution(float[] probabilities, Random random)
    {
        float sample = (float)random.NextDouble();
        float cumulative = 0f;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (sample <= cumulative)
                return i;
        }
        return probabilities.Length - 1;
    }
}
