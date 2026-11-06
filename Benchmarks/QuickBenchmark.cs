using System.Diagnostics;
using MatrixInverse.Algorithms;

namespace MatrixInverse.Benchmarks;

public static class QuickBenchmark
{
    // Розміри обрано так, щоб обчислювальна складність була достатньо великою
    // для паралелізації: від ~2*10^9 (N=1000) до ~3*10^10 (N=2500) операцій
    // множення-віднімання. Розширена матриця [A | I] займає 2*N*N*8 байт:
    // 15.3 / 34.3 / 61.0 / 95.4 МБ - тобто всі набори вже не вміщуються в L3.
    private static readonly (string Name, int Size)[] Datasets =
    {
        ("small",  1000),
        ("medium", 1500),
        ("large",  2000),
        ("xlarge", 2500),
    };

    // Кількості потоків: степені двійки + 6 (кількість P-ядер i5-12600K),
    // 10 (кількість фізичних ядер), 12 та 16 (кількість логічних потоків),
    // 32 - навмисно більше за апаратну межу.
    private static readonly int[] WorkerCounts = { 1, 2, 4, 6, 8, 10, 12, 16, 32 };
    // Прогрів JIT виконується на невеликій матриці (щоб гарячі цикли були
    // скомпільовані tier-1 JIT'ом), а не на повному наборі - це економить час.
    private const int WarmupSize = 300;
    private const int MeasureIterations = 2;
    // Пауза між запусками - щоб CPU встиг охолонути і термотроттлінг
    // не спотворював результати наступних замірів.
    private const int CooldownMs = 1500;

    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("─────────── ШВИДКИЙ БЕНЧМАРК ───────────");
        Console.WriteLine();
        Console.WriteLine($"  CPU cores:    {Environment.ProcessorCount}");
        Console.WriteLine($"  Warmup:       1 прогін на матриці {WarmupSize}x{WarmupSize}");
        Console.WriteLine($"  Measure:      {MeasureIterations} ітерацій (середнє)");
        Console.WriteLine();

        var sequentialTimes = new double[Datasets.Length];
        var parallelTimes = new double[Datasets.Length, WorkerCounts.Length];

        for (int di = 0; di < Datasets.Length; di++)
        {
            var (name, size) = Datasets[di];
            Console.Write($"  Підготовка матриці {name} (N={size})... ");
            var matrix = GetOrCreateMatrix(name, size);
            Console.WriteLine("готово.");

            Console.Write($"    sequential... ");
            sequentialTimes[di] = Measure(m => new SequentialSolver().Invert(m), matrix);
            Console.WriteLine($"{sequentialTimes[di]:F2} мс");

            for (int wi = 0; wi < WorkerCounts.Length; wi++)
            {
                int workers = WorkerCounts[wi];
                Console.Write($"    parallel (workers={workers,2})... ");
                parallelTimes[di, wi] = Measure(m => new ParallelSolver(workers).Invert(m), matrix);
                double speedup = sequentialTimes[di] / parallelTimes[di, wi];
                Console.WriteLine($"{parallelTimes[di, wi]:F2} мс  (прискорення: {speedup:F2}x)");
            }
            Console.WriteLine();
        }

        PrintMarkdownTables(sequentialTimes, parallelTimes);
        SaveCsv(sequentialTimes, parallelTimes);
    }

    private static void SaveCsv(double[] seq, double[,] par)
    {
        string testdataDir = AppPaths.FindOrCreateTestdataDir();
        string outDir = Path.GetDirectoryName(testdataDir) ?? Directory.GetCurrentDirectory();
        string csvPath = Path.Combine(outDir, "benchmark_results.csv");

        var ci = System.Globalization.CultureInfo.InvariantCulture;
        using var writer = new StreamWriter(csvPath);
        writer.Write("dataset,size,sequential_ms");
        foreach (int w in WorkerCounts)
            writer.Write($",parallel_{w}_ms,speedup_{w}");
        writer.WriteLine();

        for (int di = 0; di < Datasets.Length; di++)
        {
            var (name, size) = Datasets[di];
            writer.Write($"{name},{size},{seq[di].ToString("F4", ci)}");
            for (int wi = 0; wi < WorkerCounts.Length; wi++)
            {
                double t = par[di, wi];
                double sp = seq[di] / t;
                writer.Write($",{t.ToString("F4", ci)},{sp.ToString("F4", ci)}");
            }
            writer.WriteLine();
        }

        Console.WriteLine();
        Console.WriteLine($"  CSV з результатами збережено в: {csvPath}");
        Console.WriteLine($"  (для побудови графіків: python Scripts/plot_results.py)");
    }

    private static Matrix.Matrix GetOrCreateMatrix(string name, int size)
    {
        string testdataDir = AppPaths.FindOrCreateTestdataDir();
        string path = Path.Combine(testdataDir, $"{name}.txt");

        bool needRegenerate = true;
        if (File.Exists(path))
        {
            try
            {
                using var reader = new StreamReader(path);
                string? header = reader.ReadLine();
                if (!string.IsNullOrWhiteSpace(header) &&
                    int.TryParse(header.Trim(), out int existingN) &&
                    existingN == size)
                {
                    needRegenerate = false;
                }
            }
            catch { }
        }

        if (needRegenerate)
        {
            Matrix.Matrix.GenerateAndSave(size, path);
        }

        return Matrix.Matrix.LoadFromFile(path);
    }

    private static readonly Matrix.Matrix WarmupMatrix = Matrix.Matrix.GenerateRandom(WarmupSize, seed: 1);

    private static double Measure(Func<Matrix.Matrix, Matrix.Matrix> invert, Matrix.Matrix matrix)
    {
        // Прогрів JIT на невеликій матриці тим самим солвером.
        invert(WarmupMatrix);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Пауза для охолодження CPU перед вимірюванням.
        Thread.Sleep(CooldownMs);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < MeasureIterations; i++)
        {
            invert(matrix);
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / MeasureIterations;
    }

    private static void PrintMarkdownTables(double[] seq, double[,] par)
    {
        Console.WriteLine();
        Console.WriteLine("═══════════════ ТАБЛИЦІ ДЛЯ КУРСОВОЇ ═══════════════");
        Console.WriteLine();

        Console.WriteLine("Таблиця: Час виконання (мс)");
        Console.WriteLine();
        Console.Write("| Датасет  | Sequential ");
        foreach (int w in WorkerCounts) Console.Write($"| Parallel ({w}) ");
        Console.WriteLine("|");
        Console.Write("|---|---");
        foreach (var _ in WorkerCounts) Console.Write("|---");
        Console.WriteLine("|");
        for (int di = 0; di < Datasets.Length; di++)
        {
            Console.Write($"| {Datasets[di].Name,-8} | {seq[di]:F2} ");
            for (int wi = 0; wi < WorkerCounts.Length; wi++)
            {
                Console.Write($"| {par[di, wi]:F2} ");
            }
            Console.WriteLine("|");
        }
        Console.WriteLine();

        Console.WriteLine("Таблиця: Коефіцієнти прискорення (S = T_seq / T_par)");
        Console.WriteLine();
        Console.Write("| Датасет  ");
        foreach (int w in WorkerCounts) Console.Write($"| {w} потоки ");
        Console.WriteLine("|");
        Console.Write("|---");
        foreach (var _ in WorkerCounts) Console.Write("|---");
        Console.WriteLine("|");
        for (int di = 0; di < Datasets.Length; di++)
        {
            Console.Write($"| {Datasets[di].Name,-8} ");
            for (int wi = 0; wi < WorkerCounts.Length; wi++)
            {
                double speedup = seq[di] / par[di, wi];
                Console.Write($"| {speedup:F2}x ");
            }
            Console.WriteLine("|");
        }
        Console.WriteLine();
    }
}
