using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infrastructure.Logging;
/// <summary>
/// Simple logging interface
/// </summary>
public interface ILogger
{
    void LogInformation(string message, params object[] args);
    void LogDebug(string message, params object[] args);
    void LogWarning(string message, params object[] args);
    void LogWarning(Exception ex, string message, params object[] args);
    void LogError(Exception ex, string message, params object[] args);
}

/// <summary>
/// Generic logger interface for dependency injection
/// </summary>
public interface ILogger<T> : ILogger
{
}