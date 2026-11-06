namespace MatrixInverse.Algorithms;

/// <summary>
/// Полегшений багаторазовий бар'єр з чергуванням фаз (phase-flipping barrier)
/// на основі атомарного лічильника Interlocked та пари ManualResetEventSlim.
///
/// Останній потік, що прибув до бар'єра (Interlocked.Decrement дав 0), відновлює
/// лічильник, скидає подію НАСТУПНОЇ фази і встановлює подію ПОТОЧНОЇ фази,
/// звільняючи всіх інших. Дві події потрібні, щоб повільний потік, який ще не
/// вийшов з Wait() попередньої фази, не "проспав" сигнал: подію фази p скидають
/// лише на бар'єрі через одну фазу, коли всі потоки гарантовано вже пройшли її.
/// Set()/Wait() та Interlocked створюють повні бар'єри пам'яті, тому всі записи,
/// зроблені до бар'єра, видимі всім потокам після нього.
/// </summary>
internal sealed class PhaseBarrier : IDisposable
{
    private readonly int _participants;
    private int _remaining;
    private volatile bool _aborted;
    private readonly ManualResetEventSlim[] _events =
    {
        new ManualResetEventSlim(false),
        new ManualResetEventSlim(false),
    };

    public PhaseBarrier(int participants)
    {
        _participants = participants;
        _remaining = participants;
    }

    /// <summary>Сигналізує про прибуття та чекає на решту учасників.</summary>
    /// <param name="phase">Локальний номер фази потоку (0/1), перемикається тут.</param>
    public void SignalAndWait(ref int phase)
    {
        int p = phase;
        if (Interlocked.Decrement(ref _remaining) == 0)
        {
            Volatile.Write(ref _remaining, _participants);
            _events[p ^ 1].Reset();
            _events[p].Set();
        }
        else
        {
            // Очікування з тайм-аутом лише для того, щоб помітити аварійне
            // завершення (Abort) - у нормальному режимі Wait повертається одразу
            // після Set() останнього потоку (спершу спін, потім kernel wait).
            while (!_events[p].Wait(50))
            {
                if (_aborted) throw new OperationCanceledException("Бар'єр перервано");
            }
        }
        if (_aborted) throw new OperationCanceledException("Бар'єр перервано");
        phase = p ^ 1;
    }

    /// <summary>Аварійно звільняє всіх учасників (якщо в одному з потоків сталася помилка).</summary>
    public void Abort()
    {
        _aborted = true;
        _events[0].Set();
        _events[1].Set();
    }

    public void Dispose()
    {
        _events[0].Dispose();
        _events[1].Dispose();
    }
}
