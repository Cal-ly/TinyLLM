using ConsoleApp;

try
{
    Console.WriteLine("TinyLLM - Educational Language Model");
    Console.WriteLine("=====================================\n");

    Console.WriteLine("Select an option:");
    Console.WriteLine("1. Run Examples (test basic components)");
    Console.WriteLine("2. Train Shakespeare Model");
    Console.WriteLine("3. Exit");
    Console.Write("\nYour choice: ");

    var choice = Console.ReadLine();

    switch (choice)
    {
        case "1":
            // Run original examples
            Examples.AddSeperatorsToConsole();
            Examples.RunModelConfigurationExample();
            Examples.AddSeperatorsToConsole();
            Examples.RunMathExample();
            Examples.AddSeperatorsToConsole();
            Examples.RunTokenizerExample();
            break;

        case "2":
            // Run Shakespeare training
            await TrainShakespeare.RunTraining();
            break;

        case "3":
            Console.WriteLine("Exiting...");
            return;

        default:
            Console.WriteLine("Invalid choice!");
            break;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}

Console.WriteLine("\nPress any key to exit...");
Console.ReadKey();