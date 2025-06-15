using ConsoleApp;

try
{
    Examples.RunModelConfigurationExample();
    Examples.RunMathExample();
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
