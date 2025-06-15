using ConsoleApp;

try
{
    Examples.AddSeperatorsToConsole();
    Examples.RunModelConfigurationExample();
    Examples.AddSeperatorsToConsole();
    Examples.RunMathExample();
    Examples.AddSeperatorsToConsole();
    Examples.RunTokenizerExample();
    Console.WriteLine(new string('=', 25));
    Console.WriteLine("All examples completed successfully. Press key to end program");
    Console.ReadKey();
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
