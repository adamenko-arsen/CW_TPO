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
            ("Stress: Sequential vs Parallel agree on 200x200", TestStressAgreement),
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
        yield return new ParallelSolver(workers: 4);
        yield return new ParallelSolver(workers: 8);
    }

    private static void AssertEquivalentToIdentity(Matrix.Matrix a, Matrix.Matrix inv)
    {
        var product = Matrix.Matrix.Multiply(a, inv);
        var identity = Matrix.Matrix.Identity(a.Size);
        double err = Matrix.Matrix.MaxAbsoluteDifference(product, identity);
        if (err > Tolerance)
            throw new InvalidOperationException(
                $"||A*A^-1 - I||_inf = {err:E3} > {Tolerance:E3}");
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
        // Перший pivot = 0 - алгоритм мусить переставити рядки
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
        catch (AggregateException ae) when (ae.InnerException is InvalidOperationException io
                                            && io.Message.Contains("вироджена"))
        {
            // Parallel.For загортає виключення у AggregateException
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("вироджена"))
        {
        }
    }

    private static void TestStressAgreement()
    {
        var a = Matrix.Matrix.GenerateRandom(200, seed: 789);

        var seqResult = new SequentialSolver().Invert(a);

        foreach (int workers in new[] { 1, 2, 4, 8, 16 })
        {
            var parResult = new ParallelSolver(workers).Invert(a);
            double diff = Matrix.Matrix.MaxAbsoluteDifference(seqResult, parResult);
            if (diff > 1e-10)
                throw new InvalidOperationException(
                    $"ParallelSolver(workers={workers}) розійшовся з SequentialSolver: diff={diff:E3}");
        }
    }
}
