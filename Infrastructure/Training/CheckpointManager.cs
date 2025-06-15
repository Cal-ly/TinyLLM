using System.Text.Json;
using Core.Abstractions;
using Infrastructure.Logging;
using Infrastructure.Tokenization;

namespace Infrastructure.Training;

/// <summary>
/// Manages saving and loading model checkpoints
/// </summary>
public class CheckpointManager
{
    private readonly ILogger _logger;
    private readonly JsonSerializerOptions _jsonOptions;

    public CheckpointManager(ILogger? logger = null)
    {
        _logger = logger ?? new ConsoleLogger("CheckpointManager");
        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
    }

    public async Task SaveCheckpointAsync(
        ILanguageModel model,
        IOptimizer optimizer,
        int epoch,
        float loss,
        string filePath,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Saving checkpoint to: {Path}", filePath);

            // Ensure directory exists
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var checkpoint = new Checkpoint(
                Configuration: model.Configuration,
                ModelState: model.GetState(),
                OptimizerState: optimizer.GetState(),
                TokenizerState: null, // Would need to pass tokenizer in
                Epoch: epoch,
                Loss: loss,
                SavedAt: DateTime.UtcNow
            );

            // Serialize to JSON
            var json = JsonSerializer.Serialize(checkpoint, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json, cancellationToken);

            _logger.LogInformation("Checkpoint saved successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save checkpoint");
            throw;
        }
    }

    public async Task<Checkpoint> LoadCheckpointAsync(
        string filePath,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Loading checkpoint from: {Path}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Checkpoint file not found: {filePath}");
            }

            var json = await File.ReadAllTextAsync(filePath, cancellationToken);
            var checkpoint = JsonSerializer.Deserialize<Checkpoint>(json, _jsonOptions);

            if (checkpoint == null)
            {
                throw new InvalidOperationException("Failed to deserialize checkpoint");
            }

            _logger.LogInformation("Checkpoint loaded successfully");
            return checkpoint;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load checkpoint");
            throw;
        }
    }

    /// <summary>
    /// Find the latest checkpoint in a directory
    /// </summary>
    public string? FindLatestCheckpoint(string directory)
    {
        if (!Directory.Exists(directory))
            return null;

        var checkpoints = Directory.GetFiles(directory, "checkpoint_epoch_*.ckpt")
            .OrderByDescending(f => File.GetCreationTimeUtc(f))
            .FirstOrDefault();

        return checkpoints;
    }
}

/// <summary>
/// Extension methods for TokenizerState
/// </summary>
public static class TokenizerStateExtensions
{
    public static CharacterTokenizer ToTokenizer(this TokenizerState state)
    {
        var tokenizer = new CharacterTokenizer();
        tokenizer.LoadState(state);
        return tokenizer;
    }
}