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
            var logger = new ConsoleLogger();

            // Configuration
            // Navigate up from bin/Debug/net9.0 to solution root
            var solutionDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            var dataPath = Path.Combine(solutionDir, "Infrastructure", "Data", "shakespeare.txt");
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
                NumEpochs = 1000, // Start small for testing
                BatchSize = 16,
                SequenceLength = 64,
                ValidationSplit = 0.0f, // We already split manually
                LoggingFrequency = 50,
                CheckpointFrequency = 1,
                ValidationFrequency = 1,
                SampleGenerationFrequency = 1,
                MaxValidationBatches = 50,
                EarlyStoppingPatience = 3,
                RandomSeed = 42,
                SamplePrompts = new[] { "To be", "Romeo", "What", "The " }
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

            // Run a single training step as a test
            logger.LogInformation("\nRunning test training step...");
            await RunTestTrainingStep(context, logger);

            logger.LogInformation("\nTest completed successfully!");
            logger.LogInformation("Ready for full training loop implementation.");

            // Cleanup
            metricsLogger.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    private static async Task RunTestTrainingStep(TrainingContext context, ILogger logger)
    {
        await Task.Yield();

        // Get a single batch
        var batch = context.TrainDataset
            .CreateBatches(context.Configuration.BatchSize, context.Configuration.SequenceLength)
            .First();

        logger.LogInformation("Processing batch with {Size} sequences of length {Length}",
            batch.BatchSize, batch.SequenceLength);

        // Process first sequence as a test
        var inputTokens = batch.GetInputSequence(0);
        var targetTokens = batch.GetTargetSequence(0);

        // Forward pass
        logger.LogInformation("Running forward pass...");
        var logits = context.Model.Forward(inputTokens);
        logger.LogInformation("Forward pass complete. Output shape: [{Length}]", logits.Length);

        // Compute loss
        var targetToken = targetTokens[targetTokens.Length - 1];
        var loss = context.LossComputer.ComputeLoss(logits, targetToken);
        logger.LogInformation("Loss computed: {Loss:F4}", loss);

        // Test gradient computation
        var gradients = GradientComputations.ComputeCrossEntropyGradient(logits, targetToken);
        logger.LogInformation("Gradient computed. Shape: [{Length}]", gradients.Length);

        // Verify gradient properties
        var gradSum = gradients.Sum();
        logger.LogInformation("Gradient sum (should be ~0): {Sum:F6}", gradSum);

        // Sample the input/output
        if (context.TrainDataset.Tokenizer is CharacterTokenizer charTokenizer)
        {
            var inputText = charTokenizer.Decode(inputTokens);
            var targetChar = charTokenizer.GetCharacter(targetToken);
            var predictedToken = logits.ToList().IndexOf(logits.Max());
            var predictedChar = charTokenizer.GetCharacter(predictedToken);

            logger.LogInformation("Sample - Input: \"{Input}...\" Target: '{Target}' Predicted: '{Predicted}'",
                inputText.Length > 20 ? inputText.Substring(0, 20) : inputText,
                targetChar, predictedChar);
        }
    }
}