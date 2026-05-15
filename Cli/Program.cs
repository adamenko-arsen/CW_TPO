using System.Diagnostics;
using MatrixInverse.Algorithms;
using MatrixInverse.Benchmarks;
using MatrixInverse.Tests;

namespace MatrixInverse.Cli;

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        while (true)
        {
            try
            {
                if (!ShowMainMenu())
                    return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine($"Помилка: {ex.Message}");
                Console.WriteLine();
                Pause();
            }
        }
    }

    private static bool ShowMainMenu()
    {
        Console.Clear();
        Console.WriteLine("╔════════════════════════════════════════════╗");
        Console.WriteLine("║  Алгоритм пошуку оберненої матриці         ║");
        Console.WriteLine("║  Гаусса-Жордана, C# / .NET 8               ║");
        Console.WriteLine("║  Адаменко Арсен, ІП-35                     ║");
        Console.WriteLine("╚════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine("Головне меню:");
        Console.WriteLine("  1) Запустити алгоритм на існуючому датасеті");
        Console.WriteLine("  2) Згенерувати нову тестову матрицю");
        Console.WriteLine("  3) Запустити тести коректності");
        Console.WriteLine("  4) Запустити бенчмарк (з таблицями для курсової)");
        Console.WriteLine("  5) Згенерувати стандартний набір датасетів");
        Console.WriteLine("  0) Вийти");
        Console.WriteLine();
        Console.Write("Ваш вибір: ");

        string? choice = Console.ReadLine()?.Trim();
        Console.WriteLine();

        switch (choice)
        {
            case "0":
            case "":
            case null:
                return false;
            case "1":
                RunAlgorithm();
                Pause();
                return true;
            case "2":
                GenerateMatrix();
                Pause();
                return true;
            case "3":
                TestRunner.RunAll();
                Pause();
                return true;
            case "4":
                QuickBenchmark.Run();
                Pause();
                return true;
            case "5":
                GenerateStandardDatasets();
                Pause();
                return true;
            default:
                Console.WriteLine("Невідомий вибір. Спробуйте ще раз.");
                Pause();
                return true;
        }
    }

    private static void RunAlgorithm()
    {
        Console.WriteLine("─────────── ЗАПУСК АЛГОРИТМУ ───────────");
        Console.WriteLine();

        string testdataDir = AppPaths.FindOrCreateTestdataDir();
        var files = Directory.GetFiles(testdataDir, "*.txt")
            .OrderBy(f => new FileInfo(f).Length).ToArray();

        if (files.Length == 0)
        {
            Console.WriteLine("У директорії testdata/ немає жодної матриці.");
            Console.WriteLine("Спочатку згенеруйте матрицю (опція 2 або 5).");
            return;
        }

        Console.WriteLine("Доступні датасети:");
        for (int i = 0; i < files.Length; i++)
        {
            string name = Path.GetFileNameWithoutExtension(files[i]);
            string info = ReadHeader(files[i]);
            Console.WriteLine($"  {i + 1}) {name,-12} {info}");
        }
        Console.Write("Виберіть датасет (Enter для першого): ");
        string? input = Console.ReadLine()?.Trim();
        int dsIdx = 0;
        if (!string.IsNullOrEmpty(input))
        {
            if (!int.TryParse(input, out dsIdx) || dsIdx < 1 || dsIdx > files.Length)
            {
                Console.WriteLine("Невірний вибір. Використано перший датасет.");
                dsIdx = 1;
            }
            dsIdx--;
        }

        string datasetPath = files[dsIdx];

        Console.WriteLine();
        Console.WriteLine("Виберіть алгоритм:");
        Console.WriteLine("  1) sequential (Gauss-Jordan однопотоковий)");
        Console.WriteLine("  2) parallel (Gauss-Jordan з TPL)");
        Console.WriteLine("  3) обидва (для порівняння)");
        Console.Write("Ваш вибір (Enter для 'обидва'): ");
        string? algoChoice = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(algoChoice)) algoChoice = "3";

        int workers = 0;
        if (algoChoice == "2" || algoChoice == "3")
        {
            int defaultWorkers = Environment.ProcessorCount;
            Console.Write($"Кількість воркерів [Enter для авто = {defaultWorkers}]: ");
            string? workersInput = Console.ReadLine()?.Trim();
            if (!string.IsNullOrEmpty(workersInput) &&
                int.TryParse(workersInput, out int w) && w > 0)
            {
                workers = w;
            }
        }

        Console.WriteLine();
        Console.Write("Завантаження матриці... ");
        var matrix = Matrix.Matrix.LoadFromFile(datasetPath);
        Console.WriteLine($"готово. Розмір: {matrix.Size}x{matrix.Size}");
        Console.WriteLine();

        if (algoChoice == "1" || algoChoice == "3")
        {
            RunAndReport(new SequentialSolver(), matrix);
        }
        if (algoChoice == "2" || algoChoice == "3")
        {
            var solver = workers > 0 ? new ParallelSolver(workers) : new ParallelSolver();
            RunAndReport(solver, matrix);
        }
    }

    private static void RunAndReport(ISolver solver, Matrix.Matrix matrix)
    {
        // Прогрів JIT
        _ = solver.Invert(matrix);

        var sw = Stopwatch.StartNew();
        var inverse = solver.Invert(matrix);
        sw.Stop();

        // Швидка перевірка коректності: A * A^-1 близьке до I?
        var product = Matrix.Matrix.Multiply(matrix, inverse);
        var identity = Matrix.Matrix.Identity(matrix.Size);
        double err = Matrix.Matrix.MaxAbsoluteDifference(product, identity);

        Console.WriteLine($"  [{solver.Name}]");
        Console.WriteLine($"    Час:               {sw.Elapsed.TotalMilliseconds:F3} мс");
        Console.WriteLine($"    ||A·A⁻¹ − I||∞:    {err:E3}");
        if (solver is ParallelSolver ps)
        {
            int actualWorkers = ps.Workers > 0 ? ps.Workers : Environment.ProcessorCount;
            Console.WriteLine($"    Воркерів:          {actualWorkers}");
        }
        Console.WriteLine();
    }

    private static void GenerateMatrix()
    {
        Console.WriteLine("─────────── ГЕНЕРАЦІЯ МАТРИЦІ ───────────");
        Console.WriteLine();
        Console.Write("Розмір матриці N (Enter для 500): ");
        string? input = Console.ReadLine()?.Trim();
        int size = string.IsNullOrEmpty(input) ? 500 : int.Parse(input);

        Console.Write("Назва файлу (без розширення, Enter для 'custom'): ");
        input = Console.ReadLine()?.Trim();
        string name = string.IsNullOrEmpty(input) ? "custom" : input;

        Console.Write("Зерно ГВЧ (Enter для 42): ");
        input = Console.ReadLine()?.Trim();
        int seed = string.IsNullOrEmpty(input) ? 42 : int.Parse(input);

        string testdataDir = AppPaths.FindOrCreateTestdataDir();
        string outputPath = Path.Combine(testdataDir, $"{name}.txt");

        Console.WriteLine();
        Console.Write($"Генерую матрицю {size}x{size}... ");
        Matrix.Matrix.GenerateAndSave(size, outputPath, seed);
        Console.WriteLine("готово.");
        Console.WriteLine($"Збережено: {outputPath}");
        var fi = new FileInfo(outputPath);
        Console.WriteLine($"Розмір файлу: {FormatBytes(fi.Length)}");
    }

    private static void GenerateStandardDatasets()
    {
        Console.WriteLine("─────────── ГЕНЕРАЦІЯ СТАНДАРТНОГО НАБОРУ ───────────");
        Console.WriteLine();

        var datasets = new (string Name, int Size)[]
        {
            ("small",  400),
            ("medium", 700),
            ("large",  1000),
            ("xlarge", 1400),
        };

        string testdataDir = AppPaths.FindOrCreateTestdataDir();
        foreach (var (name, size) in datasets)
        {
            string path = Path.Combine(testdataDir, $"{name}.txt");
            Console.Write($"  {name,-7} (N={size,5})... ");
            Matrix.Matrix.GenerateAndSave(size, path);
            var fi = new FileInfo(path);
            Console.WriteLine($"готово. Розмір файлу: {FormatBytes(fi.Length)}");
        }

        Console.WriteLine();
        Console.WriteLine($"Усі датасети збережено в: {testdataDir}");
    }

    private static string ReadHeader(string path)
    {
        try
        {
            using var reader = new StreamReader(path);
            string? line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) return string.Empty;
            if (int.TryParse(line.Trim(), out int n))
                return $"(N={n})";
            return string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
        if (bytes < 1024L * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2} GB";
    }

    private static void Pause()
    {
        Console.WriteLine();
        Console.WriteLine("Натисніть Enter для повернення до меню...");
        Console.ReadLine();
    }
}
