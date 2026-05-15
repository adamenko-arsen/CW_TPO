namespace MatrixInverse.Algorithms;

public interface ISolver
{
    string Name { get; }

    /// <summary>
    /// Обчислює обернену матрицю A^(-1) для квадратної невиродженої матриці A.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Якщо матриця вироджена (singular).
    /// </exception>
    Matrix.Matrix Invert(Matrix.Matrix a);
}

public static class SolverRegistry
{
    public static IReadOnlyList<ISolver> All() =>
    [
        new SequentialSolver(),
        new ParallelSolver(),
    ];
}
