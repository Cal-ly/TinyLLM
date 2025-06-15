using System.Text;
using Infrastructure.Logging;

namespace Infrastructure.Training;

/// <summary>
/// Logs training metrics to console and optionally to file
/// </summary>
public class MetricsLogger
{
    private readonly ILogger _logger;
    private StreamWriter? _fileWriter;
    private string? _outputPath;

    public MetricsLogger(ILogger logger)
    {
        _logger = logger;
    }

    public void Initialize(TrainingConfiguration config)
    {
        // Create output directory if it doesn't exist
        Directory.CreateDirectory(config.OutputDirectory);

        // Setup metrics file
        _outputPath = Path.Combine(config.OutputDirectory, "training_metrics.csv");
        _fileWriter = new StreamWriter(_outputPath, append: false);
        _fileWriter.WriteLine("step,epoch,loss,learning_rate,gradient_norm,validation_loss,validation_perplexity");
        _fileWriter.Flush();

        _logger.LogInformation("Metrics will be saved to: {Path}", _outputPath);
    }

    public Task LogStepAsync(int step, TrainingStepResult result, CancellationToken cancellationToken)
    {
        // For now, just log to debug - could add detailed step logging if needed
        _logger.LogDebug("Step {Step}: Loss={Loss:F4}, GradNorm={GradNorm:F3}",
            step, result.Loss, result.GradientNorm);
        return Task.CompletedTask;
    }

    public Task LogEpochAsync(int epoch, EpochMetrics metrics, ValidationMetrics? validation, CancellationToken cancellationToken)
    {
        var message = new StringBuilder();
        message.AppendFormat("Epoch {0}/{1} - ", epoch + 1, "?");
        message.AppendFormat("Loss: {0:F4}, ", metrics.AverageLoss);
        message.AppendFormat("LR: {0:F6}", metrics.FinalLearningRate);

        if (validation != null)
        {
            message.AppendFormat(", Val Loss: {0:F4}, Val PPL: {1:F2}",
                validation.Loss, validation.Perplexity);
        }

        _logger.LogInformation(message.ToString());

        // Write to file
        if (_fileWriter != null)
        {
            var csvLine = $"{metrics.TotalSteps},{epoch},{metrics.AverageLoss:F6}," +
                         $"{metrics.FinalLearningRate:F8},{metrics.GradientNorm:F6}," +
                         $"{validation?.Loss ?? 0:F6},{validation?.Perplexity ?? 0:F6}";
            _fileWriter.WriteLine(csvLine);
            _fileWriter.Flush();
        }

        return Task.CompletedTask;
    }

    public void Dispose()
    {
        _fileWriter?.Dispose();
    }
}