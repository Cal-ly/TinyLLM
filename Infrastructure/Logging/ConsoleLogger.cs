using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infrastructure.Logging;
/// <summary>
/// Console logger implementation
/// </summary>
public class ConsoleLogger : ILogger
{
    private readonly string _name;

    public ConsoleLogger(string name = "")
    {
        _name = name;
    }

    public void LogInformation(string message, params object[] args)
    {
        Log("INFO", message, args);
    }

    public void LogDebug(string message, params object[] args)
    {
        Log("DEBUG", message, args);
    }

    public void LogWarning(string message, params object[] args)
    {
        Log("WARN", message, args);
    }

    public void LogWarning(Exception ex, string message, params object[] args)
    {
        Log("WARN", message, args);
        Console.WriteLine($"  Exception: {ex.Message}");
    }

    public void LogError(Exception ex, string message, params object[] args)
    {
        Log("ERROR", message, args);
        Console.WriteLine($"  Exception: {ex}");
    }

    private void Log(string level, string message, params object[] args)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        var prefix = string.IsNullOrEmpty(_name) ? "" : $"[{_name}] ";

        // Simple approach: just concatenate message with args if present
        if (args != null && args.Length > 0)
        {
            // Replace {Name} style placeholders with values in order
            var result = message;
            for (int i = 0; i < args.Length; i++)
            {
                // Find the first placeholder (either named or indexed)
                var startIndex = result.IndexOf('{');
                if (startIndex >= 0)
                {
                    var endIndex = result.IndexOf('}', startIndex);
                    if (endIndex > startIndex)
                    {
                        result = result.Substring(0, startIndex) + args[i] + result.Substring(endIndex + 1);
                    }
                }
            }
            message = result;
        }

        Console.WriteLine($"[{timestamp}] [{level}] {prefix}{message}");
    }
}

/// <summary>
/// Generic console logger implementation
/// </summary>
public class ConsoleLogger<T> : ConsoleLogger, ILogger<T>
{
    public ConsoleLogger() : base(typeof(T).Name)
    {
    }
}