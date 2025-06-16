using Core.Abstractions;
using Core.Models;
using Core.Optimizers;
using Infrastructure.Loading;
using Infrastructure.Logging;
using Infrastructure.Tokenization;
using Infrastructure.Training;

namespace ConsoleApp;

public static class TrainShakespeare
{
    public static async Task RunTraining()
    {
        Console.WriteLine("=== TinyLLM Shakespeare Training ===\n");

        try
        {
            // Setup logging
            var logFile = Path.Combine(Path.GetTempPath(), "tinyllm_training.log");
            var logger = new ConsoleLogger(logFilePath: logFile);

            // Configuration
            // Navigate up from bin/Debug/net9.0 to solution root
            var solutionDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            var dataPath = Path.Combine(solutionDir, "Infrastructure", "Data", "shakespeare25k.txt");
            var outputDir = Path.Combine(solutionDir, "Models", "shakespeare", DateTime.Now.ToString("yyyyMMdd_HHmmss"));

            // Check if data file exists
            if (!File.Exists(dataPath))
            {
                logger.LogError(null!, "Data file not found at: {Path}", dataPath);
                logger.LogInformation("Current directory: {Dir}", Environment.CurrentDirectory);
                logger.LogInformation("Looking for file at: {FullPath}", Path.GetFullPath(dataPath));
                return;
            }

            // First, load data to determine vocabulary size
            logger.LogInformation("Loading Shakespeare dataset from: {Path}", dataPath);
            var dataLoader = new TextDataLoader();
            var fullDataset = await dataLoader.LoadAsync(
                dataPath,
                preprocessingOptions: PreprocessingOptions.ForShakespeare
            );

            logger.LogInformation("Dataset loaded: {Tokens:N0} tokens, {Vocab} unique characters",
                fullDataset.TokenCount, fullDataset.Tokenizer.VocabularySize);

            // Split into train/validation
            var (trainDataset, valDataset) = fullDataset.Split(0.9f);
            logger.LogInformation("Train: {TrainTokens:N0} tokens, Val: {ValTokens:N0} tokens",
                trainDataset.TokenCount, valDataset.TokenCount);

            // Create model configuration
            var modelConfig = ModelFactory.CreateTinyModel(fullDataset.Tokenizer.VocabularySize);
            logger.LogInformation("Model configuration: {Params:N0} parameters",
                modelConfig.CalculateParameterCount());

            // Create training configuration
            var trainingConfig = new TrainingConfiguration
            {
                Name = "shakespeare-char-tiny",
                DatasetPath = dataPath,
                OutputDirectory = outputDir,
                Model = modelConfig,
                Optimizer = new OptimizerConfiguration
                {
                    Type = OptimizerTypeEnum.Adam,
                    LearningRate = 0.001f,
                    GradientClipNorm = 1.0f,
                    Schedule = new LearningRateSchedule
                    {
                        Type = ScheduleTypeEnum.Cosine,
                        WarmupSteps = 100,
                        MinLearningRate = 1e-5f
                    }
                },
                NumEpochs = 2, // Start very small for initial testing
                BatchSize = 16,
                SequenceLength = 64,
                ValidationSplit = 0.0f, // We already split manually
                LoggingFrequency = 50,
                CheckpointFrequency = 1,
                ValidationFrequency = 1,
                SampleGenerationFrequency = 1,
                SampleTemperature = 0.7f,
                MaxValidationBatches = 50,
                EarlyStoppingPatience = 3,
                RandomSeed = 42,
                SamplePrompts = new[] { "To be", "Romeo", "What", "The " },
                MetricsCsvEnabled = false
            };

            // Initialize model and optimizer
            logger.LogInformation("Initializing model and optimizer...");
            var model = new TransformerModel();
            model.Initialize(modelConfig);

            var (optimizer, scheduler) = OptimizerFactory.CreateWithScheduler(trainingConfig.Optimizer);

            // Create training components
            var metricsLogger = new MetricsLogger(logger);
            var checkpointManager = new CheckpointManager(logger);
            var validationRunner = new ValidationRunner(logger);
            var lossComputer = new LossComputer();

            // Initialize metrics logging
            metricsLogger.Initialize(trainingConfig);

            // Create training context
            var context = new TrainingContext
            {
                Model = model,
                Optimizer = optimizer,
                Scheduler = scheduler,
                LossComputer = lossComputer,
                TrainDataset = trainDataset,
                ValidationDataset = valDataset,
                Configuration = trainingConfig,
                CheckpointManager = checkpointManager,
                ValidationRunner = validationRunner
            };

            // Run training loop
            logger.LogInformation("\nStarting training...");
            var trainingLoop = new TrainingLoop(logger, metricsLogger);
            var result = await trainingLoop.RunAsync(context, startEpoch: 0, CancellationToken.None);

            logger.LogInformation("\nTraining completed!");
            logger.LogInformation("Best validation loss: {Loss:F4}", result.BestValidationLoss);
            logger.LogInformation("Total epochs: {Epochs}", result.TotalEpochs);
            logger.LogInformation("Total steps: {Steps}", result.TotalSteps);

            // Cleanup
            metricsLogger.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}