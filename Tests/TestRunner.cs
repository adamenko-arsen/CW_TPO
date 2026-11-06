using MatrixInverse.Algorithms;

namespace MatrixInverse.Tests;

public static class TestRunner
{
    private const double Tolerance = 1e-9;

    public static int RunAll()
    {
        var tests = new (string Name, Action Body)[]
        {
            ("Identity matrix 3x3", TestIdentity),
            ("Diagonal matrix", TestDiagonal),
            ("Known 2x2 matrix", TestKnown2x2),
            ("Known 3x3 matrix", TestKnown3x3),
            ("Matrix requires pivoting (zero on diagonal)", TestRequiresPivoting),
            ("Random 10x10: A * A^-1 == I", TestRandom10),
            ("Random 50x50: A * A^-1 == I", TestRandom50),
            ("Singular matrix throws (sequential)", TestSingularSequential),
            ("Singular matrix throws (parallel)", TestSingularParallel),
            ("Stress: Seq == Par, 1000x1000, 1..32 потоків", TestStressDiagDominant),
            ("Stress: Seq == Par, dense 1200x1200 з вибором pivot", TestStressPivoting),
            ("Stress: Seq == Par, N=1531 (не кратне), 3/7/10/13 потоків", TestStressOddSize),
            ("Stress: 1000x1000, A * A^-1 == I (parallel, 10 потоків)", TestStressIdentity),
            ("Repeat: 5 запусків по 16 потоків дають однаковий результат", TestRepeatability),
        };

        Console.WriteLine();
        Console.WriteLine("─────────── ТЕСТИ КОРЕКТНОСТІ ───────────");
        Console.WriteLine();

        int passed = 0, failed = 0;
        foreach (var (name, body) in tests)
        {
            try
            {
                body();
                Console.WriteLine($"  [PASS] {name}");
                passed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [FAIL] {name}");
                Console.WriteLine($"         → {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"  Підсумок: {passed} passed, {failed} failed");
        Console.WriteLine();
        return failed;
    }

    private static IEnumerable<ISolver> AllSolvers()
    {
        yield return new SequentialSolver();
        yield return new ParallelSolver(workers: 1);
        yield return new ParallelSolver(workers: 2);
        yield return new ParallelSolver(workers: 3);
        yield return new ParallelSolver(workers: 4);
        yield return new ParallelSolver(workers: 8);
        yield return new ParallelSolver(workers: 16);
    }

    private static void AssertEquivalentToIdentity(Matrix.Matrix a, Matrix.Matrix inv,
                                                   double tolerance = Tolerance)
    {
        var product = Matrix.Matrix.Multiply(a, inv);
        var identity = Matrix.Matrix.Identity(a.Size);
        double err = Matrix.Matrix.MaxAbsoluteDifference(product, identity);
        if (err > tolerance)
            throw new InvalidOperationException(
                $"||A*A^-1 - I||_inf = {err:E3} > {tolerance:E3}");
    }

    private static void TestIdentity()
    {
        foreach (var solver in AllSolvers())
        {
            var I = Matrix.Matrix.Identity(3);
            var inv = solver.Invert(I);
            double err = Matrix.Matrix.MaxAbsoluteDifference(inv, I);
            if (err > Tolerance)
                throw new InvalidOperationException(
                    $"{solver.Name}: I^-1 != I, err={err:E3}");
        }
    }

    private static void TestDiagonal()
    {
        // diag(2, 3, 5)^-1 = diag(1/2, 1/3, 1/5)
        var d = new Matrix.Matrix(3);
        d[0, 0] = 2.0; d[1, 1] = 3.0; d[2, 2] = 5.0;
        var expected = new Matrix.Matrix(3);
        expected[0, 0] = 0.5;
        expected[1, 1] = 1.0 / 3.0;
        expected[2, 2] = 0.2;

        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(d);
            double err = Matrix.Matrix.MaxAbsoluteDifference(inv, expected);
            if (err > Tolerance)
                throw new InvalidOperationException(
                    $"{solver.Name}: похибка {err:E3}");
        }
    }

    private static void TestKnown2x2()
    {
        // [[4, 7], [2, 6]]^-1 = [[0.6, -0.7], [-0.2, 0.4]]
        var a = new Matrix.Matrix(2);
        a[0, 0] = 4; a[0, 1] = 7;
        a[1, 0] = 2; a[1, 1] = 6;

        var expected = new Matrix.Matrix(2);
        expected[0, 0] = 0.6;  expected[0, 1] = -0.7;
        expected[1, 0] = -0.2; expected[1, 1] = 0.4;

        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(a);
            double err = Matrix.Matrix.MaxAbsoluteDifference(inv, expected);
            if (err > Tolerance)
                throw new InvalidOperationException(
                    $"{solver.Name}: похибка {err:E3}");
        }
    }

    private static void TestKnown3x3()
    {
        // Перевіряємо через A * A^-1 = I, бо тут аналітичну відповідь
        // утомливо вписувати, але можемо вірити в основну тотожність.
        var a = new Matrix.Matrix(3);
        a[0, 0] = 1; a[0, 1] = 2; a[0, 2] = 3;
        a[1, 0] = 0; a[1, 1] = 1; a[1, 2] = 4;
        a[2, 0] = 5; a[2, 1] = 6; a[2, 2] = 0;

        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(a);
            AssertEquivalentToIdentity(a, inv);
        }
    }

    private static void TestRequiresPivoting()
    {
        // Перший pivot = 0 - алгоритм мусить підсилити опорний рядок,
        // додавши до нього рядок з найбільшим за модулем елементом стовпця
        var a = new Matrix.Matrix(3);
        a[0, 0] = 0; a[0, 1] = 1; a[0, 2] = 2;
        a[1, 0] = 1; a[1, 1] = 0; a[1, 2] = 3;
        a[2, 0] = 4; a[2, 1] = 5; a[2, 2] = 0;

        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(a);
            AssertEquivalentToIdentity(a, inv);
        }
    }

    private static void TestRandom10()
    {
        var a = Matrix.Matrix.GenerateRandom(10, seed: 123);
        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(a);
            AssertEquivalentToIdentity(a, inv);
        }
    }

    private static void TestRandom50()
    {
        var a = Matrix.Matrix.GenerateRandom(50, seed: 456);
        foreach (var solver in AllSolvers())
        {
            var inv = solver.Invert(a);
            AssertEquivalentToIdentity(a, inv);
        }
    }

    private static void TestSingularSequential()
    {
        // Рядок 1 = 2 * рядок 0 - матриця вироджена
        var a = new Matrix.Matrix(3);
        a[0, 0] = 1; a[0, 1] = 2; a[0, 2] = 3;
        a[1, 0] = 2; a[1, 1] = 4; a[1, 2] = 6;
        a[2, 0] = 7; a[2, 1] = 8; a[2, 2] = 9;

        try
        {
            new SequentialSolver().Invert(a);
            throw new InvalidOperationException("Очікувалось виключення для виродженої матриці");
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("вироджена"))
        {
            // Очікувана поведінка
        }
    }

    private static void TestSingularParallel()
    {
        var a = new Matrix.Matrix(3);
        a[0, 0] = 1; a[0, 1] = 2; a[0, 2] = 3;
        a[1, 0] = 2; a[1, 1] = 4; a[1, 2] = 6;
        a[2, 0] = 7; a[2, 1] = 8; a[2, 2] = 9;

        try
        {
            new ParallelSolver(workers: 4).Invert(a);
            throw new InvalidOperationException("Очікувалось виключення");
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("вироджена"))
        {
            // Очікувана поведінка: усі потоки пулу однаково виявили нульовий
            // головний елемент, коректно завершились, а виняток кинуто з Invert.
        }
    }

    // ───────────── Стрес-тести паралельної версії на великих матрицях ─────────────
    // Розміри обрано так, щоб на кожен потік припадали сотні рядків, а фаза
    // виключення тривала мілісекунди: конфлікти потоків (гонки, неповна
    // синхронізація фаз, невидимість записів) за таких умов неминуче проявили б
    // себе. Послідовна та паралельна версії виконують ідентичні арифметичні
    // операції в однаковому порядку, тому за відсутності гонок результати
    // мають збігатися ПОБІТОВО (різниця рівно 0).

    private static readonly int[] StressWorkers = { 1, 2, 3, 4, 6, 8, 10, 12, 16, 32 };

    private static void AssertSameAsSequential(Matrix.Matrix a, IEnumerable<int> workerCounts)
    {
        var seqResult = new SequentialSolver().Invert(a);
        foreach (int workers in workerCounts)
        {
            var parResult = new ParallelSolver(workers).Invert(a);
            double diff = Matrix.Matrix.MaxAbsoluteDifference(seqResult, parResult);
            if (diff != 0.0)
                throw new InvalidOperationException(
                    $"N={a.Size}, workers={workers}: результат відрізняється від послідовного, diff={diff:E3}");
        }
    }

    private static void TestStressDiagDominant()
    {
        var a = Matrix.Matrix.GenerateRandom(1000, seed: 789);
        AssertSameAsSequential(a, StressWorkers);
    }

    private static void TestStressPivoting()
    {
        // Щільна випадкова матриця без діагонального домінування: головний
        // елемент майже на кожній ітерації знаходиться в іншому рядку, тож
        // постійно працює гілка підсилення опорного рядка (читання чужого рядка r).
        var a = Matrix.Matrix.GenerateRandomDense(1200, seed: 2024);
        AssertSameAsSequential(a, new[] { 2, 4, 8, 10, 16 });
        AssertEquivalentToIdentity(a, new ParallelSolver(10).Invert(a), 1e-8);
    }

    private static void TestStressOddSize()
    {
        // 1531 - просте число: смуги рядків і стовпців мають різну довжину,
        // останній потік отримує неповну смугу.
        var a = Matrix.Matrix.GenerateRandom(1531, seed: 31);
        AssertSameAsSequential(a, new[] { 3, 7, 10, 13 });
    }

    private static void TestStressIdentity()
    {
        var a = Matrix.Matrix.GenerateRandom(1000, seed: 1000);
        var inv = new ParallelSolver(10).Invert(a);
        AssertEquivalentToIdentity(a, inv);
    }

    private static void TestRepeatability()
    {
        var a = Matrix.Matrix.GenerateRandomDense(1000, seed: 77);
        var first = new ParallelSolver(16).Invert(a);
        for (int run = 1; run < 5; run++)
        {
            var again = new ParallelSolver(16).Invert(a);
            double diff = Matrix.Matrix.MaxAbsoluteDifference(first, again);
            if (diff != 0.0)
                throw new InvalidOperationException(
                    $"Запуск {run + 1} відрізняється від першого: diff={diff:E3}");
        }
    }
}
