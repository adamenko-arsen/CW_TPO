namespace MatrixInverse;

internal static class AppPaths
{
    /// <summary>
    /// Знаходить або створює директорію testdata/ у корені проєкту.
    /// Пошук іде вгору від поточної робочої директорії - це дозволяє
    /// запускати програму як з bin/Debug/net8.0/, так і з корня репозиторію.
    /// </summary>
    public static string FindOrCreateTestdataDir()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "testdata");
            if (Directory.Exists(candidate)) return candidate;
            if (File.Exists(Path.Combine(dir.FullName, "MatrixInverse.csproj")))
            {
                Directory.CreateDirectory(candidate);
                return candidate;
            }
            dir = dir.Parent;
        }
        string fallback = Path.Combine(Directory.GetCurrentDirectory(), "testdata");
        Directory.CreateDirectory(fallback);
        return fallback;
    }
}
