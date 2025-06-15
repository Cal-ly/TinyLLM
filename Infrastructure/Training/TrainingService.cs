using Core.Abstractions;
using Core.Models;
using Core.Optimizers;
using Infrastructure.Loading;
using Infrastructure.Logging;

namespace Infrastructure.Training;

/// <summary>
/// Main training service that orchestrates the complete training pipeline
/// Brings together model, optimizer, data, and training loop
/// </summary>
public sealed class TrainingService
{
    private readonly ILogger<TrainingService> _logger;
    private readonly MetricsLogger _metricsLogger;
    private readonly CheckpointManager _checkpointManager;
    private readonly ValidationRunner _validationRunner;

    public TrainingService(
        ILogger<TrainingService> logger,
        MetricsLogger metricsLogger,
        CheckpointManager checkpointManager,
        ValidationRunner validationRunner)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _metricsLogger = metricsLogger ?? throw new ArgumentNullException(nameof(metricsLogger));
        _checkpointManager = checkpointManager ?? throw new ArgumentNullException(nameof(checkpointManager));
        _validationRunner = validationRunner ?? throw new ArgumentNullException(nameof(validationRunner));
    }

    /// <summary>
    /// Train a model with the specified configuration
    /// </summary>
    public async Task<TrainingResult> TrainAsync(
        TrainingConfiguration config,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting training: {ConfigName}", config.Name);
        _logger.LogInformation("Model: {ParameterCount:N0} parameters", config.Model.CalculateParameterCount());

        try
        {
            // Validate configuration
            ValidateConfiguration(config);

            // Load and prepare data
            var (trainDataset, validationDataset) = await LoadDataAsync(config, cancellationToken);
            _logger.LogInformation("Loaded training data: {TrainTokens:N0} tokens, validation: {ValTokens:N0} tokens",
                trainDataset.TokenCount, validationDataset?.TokenCount ?? 0);

            // Initialize model and optimizer
            var model = CreateModel(config.Model); // Pass config.Model, not config
            var (optimizer, scheduler) = CreateOptimizer(config);

            // Setup training components
            var trainingLoop = new TrainingLoop(_logger, _metricsLogger);
            var lossComputer = new LossComputer();

            // Initialize metrics logging
            _metricsLogger.Initialize(config);

            // Resume from checkpoint if specified
            int startEpoch = 0;
            if (!string.IsNullOrEmpty(config.ResumeFromCheckpoint))
            {
                startEpoch = await ResumeFromCheckpointAsync(config.ResumeFromCheckpoint, model, optimizer);
                _logger.LogInformation("Resumed training from epoch {Epoch}", startEpoch);
            }

            // Run training loop
            var trainingContext = new TrainingContext
            {
                Model = model,
                Optimizer = optimizer,
                Scheduler = scheduler,
                LossComputer = lossComputer,
                TrainDataset = trainDataset,
                ValidationDataset = validationDataset,
                Configuration = config,
                CheckpointManager = _checkpointManager,
                ValidationRunner = _validationRunner
            };

            var result = await trainingLoop.RunAsync(trainingContext, startEpoch, cancellationToken);

            _logger.LogInformation("Training completed successfully");
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Training failed");
            throw;
        }
    }

    /// <summary>
    /// Evaluate a trained model on a dataset
    /// </summary>
    public async Task<EvaluationResult> EvaluateAsync(
        string modelPath,
        string datasetPath,
        EvaluationConfiguration config,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting model evaluation");

        // Load model
        var checkpoint = await _checkpointManager.LoadCheckpointAsync(modelPath, cancellationToken);
        var model = CreateModel(checkpoint.Configuration);
        model.LoadState(checkpoint.ModelState);

        // Load evaluation dataset
        var dataLoader = new TextDataLoader();
        var dataset = await dataLoader.LoadAsync(datasetPath, checkpoint.TokenizerState?.ToTokenizer(), cancellationToken: cancellationToken);

        // Run evaluation
        return await _validationRunner.EvaluateAsync(model, dataset, config, cancellationToken);
    }

    private void ValidateConfiguration(TrainingConfiguration config)
    {
        if (string.IsNullOrWhiteSpace(config.Name))
            throw new ArgumentException("Training configuration must have a name");

        if (string.IsNullOrWhiteSpace(config.DatasetPath))
            throw new ArgumentException("Dataset path is required");

        if (string.IsNullOrWhiteSpace(config.OutputDirectory))
            throw new ArgumentException("Output directory is required");

        config.Model.Validate();

        var optimizerValidation = OptimizerFactory.ValidateConfiguration(config.Optimizer);
        if (!optimizerValidation.IsValid)
        {
            throw new ArgumentException($"Invalid optimizer configuration: {string.Join(", ", optimizerValidation.Errors)}");
        }
    }

    private async Task<(TextDataset train, TextDataset? validation)> LoadDataAsync(
        TrainingConfiguration config,
        CancellationToken cancellationToken)
    {
        var dataLoader = new TextDataLoader();
        var dataset = await dataLoader.LoadAsync(config.DatasetPath, cancellationToken: cancellationToken);

        if (config.ValidationSplit > 0f)
        {
            var (train, validation) = dataset.Split(1f - config.ValidationSplit, config.RandomSeed);
            return (train, validation);
        }

        return (dataset, null);
    }

    private static ILanguageModel CreateModel(Core.Abstractions.ModelConfiguration modelConfig)
    {
        var model = new TransformerModel();
        model.Initialize(modelConfig);
        return model;
    }

    private static (IOptimizer optimizer, LearningRateScheduler scheduler) CreateOptimizer(TrainingConfiguration config)
    {
        return OptimizerFactory.CreateWithScheduler(config.Optimizer);
    }

    private async Task<int> ResumeFromCheckpointAsync(string checkpointPath, ILanguageModel model, IOptimizer optimizer)
    {
        var checkpoint = await _checkpointManager.LoadCheckpointAsync(checkpointPath);

        model.LoadState(checkpoint.ModelState);
        optimizer.LoadState(checkpoint.OptimizerState);

        return checkpoint.Epoch;
    }
}