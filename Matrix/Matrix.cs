using System.Globalization;

namespace MatrixInverse.Matrix;

/// <summary>
/// Квадратна матриця double, збережена у row-major порядку у плоскому масиві
/// для кращої cache-locality та можливості SIMD-операцій у майбутньому.
/// </summary>
public sealed class Matrix
{
    public int Size { get; }
    public double[] Data { get; }

    public Matrix(int size)
    {
        if (size <= 0) throw new ArgumentException("Розмір має бути > 0", nameof(size));
        Size = size;
        Data = new double[(long)size * size];
    }

    public double this[int row, int col]
    {
        get => Data[(long)row * Size + col];
        set => Data[(long)row * Size + col] = value;
    }

    public Matrix Clone()
    {
        var copy = new Matrix(Size);
        Array.Copy(Data, copy.Data, Data.Length);
        return copy;
    }

    public static Matrix Identity(int n)
    {
        var m = new Matrix(n);
        for (int i = 0; i < n; i++) m[i, i] = 1.0;
        return m;
    }

    /// <summary>
    /// Генерує випадкову, гарантовано невироджену матрицю розміру N.
    /// Метод: A = R + N*I, де R - випадкові числа у [-1, 1], а діагональне домінування
    /// (N на діагоналі) гарантує умовно високий cond(A) і відсутність виродженості.
    /// </summary>
    public static Matrix GenerateRandom(int n, int seed = 42)
    {
        var rng = new Random(seed);
        var m = new Matrix(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                m[i, j] = rng.NextDouble() * 2.0 - 1.0;
            }
            // Діагональне домінування - гарантує невиродженість
            m[i, i] += n;
        }
        return m;
    }

    /// <summary>
    /// Множення матриць A * B. Використовується для верифікації: A * A^(-1) = I.
    /// </summary>
    /// <summary>
    /// Щільна випадкова матриця з елементами у [-1, 1) БЕЗ діагонального домінування.
    /// Використовується у стрес-тестах, щоб вибір головного елемента реально
    /// спрацьовував (найбільший елемент стовпця здебільшого не на діагоналі).
    /// </summary>
    public static Matrix GenerateRandomDense(int n, int seed = 42)
    {
        var rng = new Random(seed);
        var m = new Matrix(n);
        for (long i = 0; i < m.Data.Length; i++)
            m.Data[i] = rng.NextDouble() * 2.0 - 1.0;
        return m;
    }

    public static Matrix Multiply(Matrix a, Matrix b)
    {
        if (a.Size != b.Size)
            throw new ArgumentException("Розміри матриць не співпадають");
        int n = a.Size;
        var c = new Matrix(n);
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                double aik = a[i, k];
                if (aik == 0.0) continue;
                for (int j = 0; j < n; j++)
                {
                    c[i, j] += aik * b[k, j];
                }
            }
        }
        return c;
    }

    /// <summary>
    /// Максимальна абсолютна різниця між матрицями (∞-норма).
    /// Використовується для перевірки, наскільки A*A^(-1) близьке до I.
    /// </summary>
    public static double MaxAbsoluteDifference(Matrix a, Matrix b)
    {
        if (a.Size != b.Size) throw new ArgumentException("Розміри не співпадають");
        double max = 0.0;
        for (long i = 0; i < a.Data.Length; i++)
        {
            double diff = Math.Abs(a.Data[i] - b.Data[i]);
            if (diff > max) max = diff;
        }
        return max;
    }

    public static Matrix LoadFromFile(string path)
    {
        using var reader = new StreamReader(path);
        var header = reader.ReadLine()!.Trim();
        int n = int.Parse(header, CultureInfo.InvariantCulture);
        var m = new Matrix(n);
        for (int i = 0; i < n; i++)
        {
            var line = reader.ReadLine()
                ?? throw new InvalidDataException($"Очікувався рядок {i + 1}");
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != n)
                throw new InvalidDataException(
                    $"Рядок {i + 1}: очікувалось {n} значень, отримано {parts.Length}");
            for (int j = 0; j < n; j++)
            {
                m[i, j] = double.Parse(parts[j], CultureInfo.InvariantCulture);
            }
        }
        return m;
    }

    public void SaveToFile(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine(Size);
        var ci = CultureInfo.InvariantCulture;
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                if (j > 0) writer.Write(' ');
                writer.Write(this[i, j].ToString("G17", ci));
            }
            writer.WriteLine();
        }
    }

    public static void GenerateAndSave(int n, string path, int seed = 42)
    {
        var m = GenerateRandom(n, seed);
        m.SaveToFile(path);
    }
}
