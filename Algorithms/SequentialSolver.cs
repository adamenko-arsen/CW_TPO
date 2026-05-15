namespace MatrixInverse.Algorithms;

/// <summary>
/// Послідовна реалізація алгоритму Гаусса-Жордана для обернення матриці.
/// Працює з розширеною матрицею [A | I], яка зберігається у плоскому масиві
/// розміру N x 2N для cache-friendly доступу. Застосовується частковий вибір
/// головного елемента (partial pivoting) для чисельної стабільності.
/// </summary>
public sealed class SequentialSolver : ISolver
{
    public string Name => "sequential";

    public Matrix.Matrix Invert(Matrix.Matrix a)
    {
        int n = a.Size;
        // Розширена матриця [A | I] зберігається як плоский row-major масив
        // довжиною N*(2N). aug[i*W + j] = елемент у рядку i, стовпці j, де W = 2N.
        int w = 2 * n;
        var aug = new double[(long)n * w];

        // Ініціалізація: лівa половина = A, правa половина = I
        for (int i = 0; i < n; i++)
        {
            long baseIdx = (long)i * w;
            for (int j = 0; j < n; j++)
            {
                aug[baseIdx + j] = a[i, j];
            }
            aug[baseIdx + n + i] = 1.0;
        }

        // Прямий + зворотний хід Гаусса-Жордана з частковим pivoting'ом
        for (int k = 0; k < n; k++)
        {
            // 1) Вибір головного елемента: знаходимо рядок з max |a[r,k]| у [k..n-1]
            int pivotRow = k;
            double pivotVal = Math.Abs(aug[(long)k * w + k]);
            for (int r = k + 1; r < n; r++)
            {
                double v = Math.Abs(aug[(long)r * w + k]);
                if (v > pivotVal)
                {
                    pivotVal = v;
                    pivotRow = r;
                }
            }

            if (pivotVal < 1e-15)
                throw new InvalidOperationException(
                    $"Матриця вироджена або погано обумовлена: pivot[{k}] = {pivotVal}");

            // 2) Перестановка рядків k <-> pivotRow (якщо потрібно)
            if (pivotRow != k)
            {
                long ofsK = (long)k * w;
                long ofsP = (long)pivotRow * w;
                for (int j = 0; j < w; j++)
                {
                    (aug[ofsK + j], aug[ofsP + j]) = (aug[ofsP + j], aug[ofsK + j]);
                }
            }

            // 3) Нормалізація опорного рядка: ділимо весь рядок на pivot
            long pivotOfs = (long)k * w;
            double pivot = aug[pivotOfs + k];
            double invPivot = 1.0 / pivot;
            for (int j = 0; j < w; j++)
            {
                aug[pivotOfs + j] *= invPivot;
            }

            // 4) Виключення: для всіх інших рядків i != k віднімаємо aug[i,k] * pivotRow
            for (int i = 0; i < n; i++)
            {
                if (i == k) continue;
                long rowOfs = (long)i * w;
                double factor = aug[rowOfs + k];
                if (factor == 0.0) continue;
                for (int j = 0; j < w; j++)
                {
                    aug[rowOfs + j] -= factor * aug[pivotOfs + j];
                }
            }
        }

        // Витягуємо праву половину - це і є A^(-1)
        var result = new Matrix.Matrix(n);
        for (int i = 0; i < n; i++)
        {
            long baseIdx = (long)i * w + n;
            for (int j = 0; j < n; j++)
            {
                result[i, j] = aug[baseIdx + j];
            }
        }
        return result;
    }
}
