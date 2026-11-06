namespace MatrixInverse.Algorithms;

/// <summary>
/// Паралельна реалізація алгоритму Гаусса-Жордана з персистентним пулом потоків
/// та полегшеною бар'єрною синхронізацією (PhaseBarrier: Interlocked + пара
/// ManualResetEventSlim).
///
/// Усі етапи алгоритму виконуються паралельно, кожен потік t працює зі своєю
/// частиною даних, яку визначає його індекс t:
///   - смуга рядків   [t*rowChunk, (t+1)*rowChunk)  - ініціалізація [A | I],
///     виключення, пошук локального головного елемента, копіювання результату;
///   - смуга стовпців [t*colChunk, (t+1)*colChunk) - підсилення та нормалізація
///     опорного рядка.
///
/// Одна ітерація k містить дві фази, розділені бар'єрами:
///   Фаза 1 (стовпці): кожен потік редукує локальні максимуми всіх потоків
///          (однаковий детермінований результат у кожному потоці) і формує свою
///          частину нормалізованого опорного рядка
///          aug[k][j] := (aug[k][j] + s*aug[r][j]) / p   для j зі своєї смуги.
///   Фаза 2 (рядки): кожен потік виключає стовпець k зі своїх рядків і одразу
///          (злито з виключенням) шукає локальний максимум |aug[i][k+1]| для
///          наступної ітерації.
/// Головний потік сам є учасником пулу (t = 0), тому окремої послідовної
/// частини між фазами немає.
/// </summary>
public sealed class ParallelSolver : ISolver
{
    public string Name => "parallel";
    public int Workers { get; }

    public ParallelSolver() : this(0) { }
    public ParallelSolver(int workers) { Workers = workers; }

    public Matrix.Matrix Invert(Matrix.Matrix a)
    {
        int workers = Workers > 0 ? Workers : Environment.ProcessorCount;
        var job = new Job(a, workers);

        // Пул: workers-1 додаткових потоків + поточний (головний) потік як t = 0.
        var threads = new Thread[workers - 1];
        for (int t = 1; t < workers; t++)
        {
            int threadIndex = t;
            threads[t - 1] = new Thread(() => job.Worker(threadIndex))
            {
                IsBackground = true,
                Name = $"GaussJordan-Worker-{threadIndex}"
            };
            threads[t - 1].Start();
        }

        job.Worker(0);
        foreach (var th in threads) th.Join();
        job.Barrier.Dispose();

        if (job.Error != null) throw job.Error;
        if (job.SingularStep >= 0)
            throw new InvalidOperationException(
                $"Матриця вироджена або погано обумовлена: pivot[{job.SingularStep}] = {job.SingularValue}");
        return job.Result;
    }

    /// <summary>Спільний стан однієї операції обернення.</summary>
    private sealed class Job
    {
        private readonly Matrix.Matrix _a;
        private readonly int _n, _w, _workers, _rowChunk, _colChunk;
        private readonly double[] _aug;

        // Локальні результати пошуку головного елемента (по одному на потік).
        private readonly double[] _localMax;   // max |aug[i][k]| серед своїх рядків i >= k
        private readonly int[] _localRow;      // номер рядка з цим максимумом
        private readonly double[] _localVal;   // значення aug[i][k] (зі знаком)
        private readonly double[] _diag;       // aug[k][k], пише лише власник рядка k

        public readonly PhaseBarrier Barrier;
        public readonly Matrix.Matrix Result;
        public volatile Exception? Error;
        public int SingularStep = -1;
        public double SingularValue;

        public Job(Matrix.Matrix a, int workers)
        {
            _a = a;
            _n = a.Size;
            _w = 2 * _n;
            _workers = workers;
            _rowChunk = (_n + workers - 1) / workers;
            _colChunk = (_w + workers - 1) / workers;
            _aug = new double[(long)_n * _w];
            _localMax = new double[workers];
            _localRow = new int[workers];
            _localVal = new double[workers];
            _diag = new double[workers];
            Barrier = new PhaseBarrier(workers);
            Result = new Matrix.Matrix(_n);
        }

        public void Worker(int t)
        {
            int n = _n, w = _w;
            double[] aug = _aug;
            int rowFrom = Math.Min(n, t * _rowChunk);
            int rowTo = Math.Min(n, rowFrom + _rowChunk);
            int colFrom = Math.Min(w, t * _colChunk);
            int colTo = Math.Min(w, colFrom + _colChunk);
            int phase = 0;

            try
            {
                // 0) Паралельна ініціалізація своїх рядків [A | I] + пошук pivot для k = 0.
                double[] src = _a.Data;
                for (int i = rowFrom; i < rowTo; i++)
                {
                    long rowOfs = (long)i * w;
                    Array.Copy(src, (long)i * n, aug, rowOfs, n);
                    aug[rowOfs + n + i] = 1.0;
                }
                ScanColumn(t, 0, rowFrom, rowTo);
                Barrier.SignalAndWait(ref phase);

                for (int k = 0; k < n; k++)
                {
                    // 1) Редукція локальних максимумів (однаково в усіх потоках).
                    int pivotRow = -1;
                    double pivotVal = -1.0, ark = 0.0;
                    for (int u = 0; u < _workers; u++)
                    {
                        if (_localMax[u] > pivotVal)
                        {
                            pivotVal = _localMax[u];
                            pivotRow = _localRow[u];
                            ark = _localVal[u];
                        }
                    }
                    if (pivotVal < SequentialSolver.SingularEps)
                    {
                        if (t == 0) { SingularStep = k; SingularValue = Math.Max(pivotVal, 0.0); }
                        break; // усі потоки бачать те саме значення і виходять разом
                    }

                    // 2) Своя смуга стовпців опорного рядка: підсилення + нормалізація.
                    long pivotOfs = (long)k * w;
                    double akk = _diag[k / _rowChunk];
                    if (pivotRow != k)
                    {
                        long ofsR = (long)pivotRow * w;
                        double s = akk * ark >= 0.0 ? 1.0 : -1.0;
                        double invPivot = 1.0 / (akk + s * ark);
                        for (int j = colFrom; j < colTo; j++)
                            aug[pivotOfs + j] = (aug[pivotOfs + j] + s * aug[ofsR + j]) * invPivot;
                    }
                    else
                    {
                        double invPivot = 1.0 / akk;
                        for (int j = colFrom; j < colTo; j++)
                            aug[pivotOfs + j] *= invPivot;
                    }
                    Barrier.SignalAndWait(ref phase);

                    // 3) Виключення у своїх рядках + пошук pivot для k + 1 (злито).
                    int next = k + 1;
                    double best = -1.0, bestVal = 0.0;
                    int bestRow = -1;
                    for (int i = rowFrom; i < rowTo; i++)
                    {
                        if (i == k) continue;
                        long rowOfs = (long)i * w;
                        double factor = aug[rowOfs + k];
                        if (factor != 0.0)
                        {
                            for (int j = 0; j < w; j++)
                                aug[rowOfs + j] -= factor * aug[pivotOfs + j];
                        }
                        if (i >= next && next < n)
                        {
                            double v = aug[rowOfs + next];
                            if (i == next) _diag[t] = v;
                            double av = Math.Abs(v);
                            if (av > best) { best = av; bestRow = i; bestVal = v; }
                        }
                    }
                    _localMax[t] = best;
                    _localRow[t] = bestRow;
                    _localVal[t] = bestVal;
                    Barrier.SignalAndWait(ref phase);
                }

                // 4) Паралельне копіювання своїх рядків правої половини у результат.
                if (SingularStep < 0)
                {
                    double[] dst = Result.Data;
                    for (int i = rowFrom; i < rowTo; i++)
                        Array.Copy(aug, (long)i * w + n, dst, (long)i * n, n);
                }
            }
            catch (OperationCanceledException) when (Error != null)
            {
                // Інший потік уже зафіксував помилку і перервав бар'єр.
            }
            catch (Exception ex)
            {
                Error ??= ex;
                Barrier.Abort();
            }
        }

        /// <summary>Локальний пошук max |aug[i][col]| серед своїх рядків i >= col.</summary>
        private void ScanColumn(int t, int col, int rowFrom, int rowTo)
        {
            double best = -1.0, bestVal = 0.0;
            int bestRow = -1;
            for (int i = Math.Max(rowFrom, col); i < rowTo; i++)
            {
                double v = _aug[(long)i * _w + col];
                if (i == col) _diag[t] = v;
                double av = Math.Abs(v);
                if (av > best) { best = av; bestRow = i; bestVal = v; }
            }
            _localMax[t] = best;
            _localRow[t] = bestRow;
            _localVal[t] = bestVal;
        }
    }
}
