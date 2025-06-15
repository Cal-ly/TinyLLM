using ConsoleApp;

try
{
    Console.WriteLine("TinyLLM - Educational Language Model");
    Console.WriteLine("=====================================\n");

    Console.WriteLine("Select an option:");
    Console.WriteLine("1. Train Shakespeare Model");
    Console.WriteLine("2. Exit");
    Console.Write("\nYour choice: ");

    var choice = Console.ReadLine();

    switch (choice)
    {
        case "1":
            // Run Shakespeare training
            await TrainShakespeare.RunTraining();
            break;

        case "2":
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