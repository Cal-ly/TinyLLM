using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core.Abstractions;
/// <summary>
/// Configuration for text generation
/// </summary>
public record GenerationConfiguration
{
    public required GenerationStrategy Strategy { get; init; }
    public int MaxTokens { get; init; } = 100;
    public float Temperature { get; init; } = 0.8f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 0.9f;
    public string[] StopTokens { get; init; } = [];
    public bool Echo { get; init; } = true; // Include prompt in output
    public int? Seed { get; init; } // For reproducible generation
}

public enum GenerationStrategy
{
    Greedy,
    Temperature,
    TopK,
    TopP,
    BeamSearch
}

/// <summary>
/// Result of text generation
/// </summary>
public record GenerationResult(
    string GeneratedText,
    string FullText, // Prompt + generated
    int TokensGenerated,
    TimeSpan GenerationTime,
    string StopReason // "max_tokens", "stop_token", "end_token"
)
{
    public float TokensPerSecond => TokensGenerated / (float)GenerationTime.TotalSeconds;
}