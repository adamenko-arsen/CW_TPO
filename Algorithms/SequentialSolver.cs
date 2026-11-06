namespace MatrixInverse.Algorithms;

/// <summary>
/// Послідовна (однопотокова) реалізація обернення матриці методом Гаусса-Жордана
/// над розширеною матрицею [A | I].
///
/// Вибір головного елемента виконується БЕЗ перестановки рядків: якщо у стовпці k
/// нижче діагоналі знайдено більший за модулем елемент (рядок r), то рядок r
/// додається до опорного рядка k зі знаком, при якому модулі складаються:
///     aug[k] := aug[k] + s * aug[r],   s = sign(aug[k][k] * aug[r][k]).
/// Це елементарне перетворення рядків, тому права половина в кінці дорівнює A^(-1)
/// без жодних зворотних перестановок, а порядок рядків матриці не змінюється.
/// Новий опорний елемент |aug[k][k]| = |a_kk| + |a_rk| >= max |a_ik|.
/// </summary>
public sealed class SequentialSolver : ISolver
{
    /// <summary>Поріг, нижче якого опорний елемент вважається нульовим.</summary>
    public const double SingularEps = 1e-15;

    public string Name => "sequential";

    public Matrix.Matrix Invert(Matrix.Matrix a)
    {
        int n = a.Size;
        int w = 2 * n;
        var aug = new double[(long)n * w];

        // Ініціалізація розширеної матриці [A | I]
        for (int i = 0; i < n; i++)
        {
            long baseIdx = (long)i * w;
            for (int j = 0; j < n; j++)
                aug[baseIdx + j] = a[i, j];
            aug[baseIdx + n + i] = 1.0;
        }

        for (int k = 0; k < n; k++)
        {
            long pivotOfs = (long)k * w;

            // 1) Пошук головного елемента у стовпці k серед рядків k..n-1
            int pivotRow = k;
            double pivotVal = Math.Abs(aug[pivotOfs + k]);
            for (int r = k + 1; r < n; r++)
            {
                double v = Math.Abs(aug[(long)r * w + k]);
                if (v > pivotVal) { pivotVal = v; pivotRow = r; }
            }
            if (pivotVal < SingularEps)
                throw new InvalidOperationException(
                    $"Матриця вироджена або погано обумовлена: pivot[{k}] = {pivotVal}");

            // 2) Підсилення опорного рядка (без перестановки рядків) + нормалізація
            double akk = aug[pivotOfs + k];
            if (pivotRow != k)
            {
                long ofsR = (long)pivotRow * w;
                double ark = aug[ofsR + k];
                double s = akk * ark >= 0.0 ? 1.0 : -1.0;
                double invPivot = 1.0 / (akk + s * ark);
                for (int j = 0; j < w; j++)
                    aug[pivotOfs + j] = (aug[pivotOfs + j] + s * aug[ofsR + j]) * invPivot;
            }
            else
            {
                double invPivot = 1.0 / akk;
                for (int j = 0; j < w; j++)
                    aug[pivotOfs + j] *= invPivot;
            }

            // 3) Виключення стовпця k з усіх інших рядків - послідовно
            for (int i = 0; i < n; i++)
            {
                if (i == k) continue;
                long rowOfs = (long)i * w;
                double factor = aug[rowOfs + k];
                if (factor == 0.0) continue;
                for (int j = 0; j < w; j++)
                    aug[rowOfs + j] -= factor * aug[pivotOfs + j];
            }
        }

        // Права половина розширеної матриці = A^(-1)
        var result = new Matrix.Matrix(n);
        for (int i = 0; i < n; i++)
        {
            long baseIdx = (long)i * w + n;
            for (int j = 0; j < n; j++)
                result[i, j] = aug[baseIdx + j];
        }
        return result;
    }
}
