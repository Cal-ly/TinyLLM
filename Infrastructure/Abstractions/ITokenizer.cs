namespace Infrastructure.Abstractions;

public interface ITokenizer
{
    int VocabularySize { get; }
    string Decode(IEnumerable<int> tokens);
    int[] Encode(ReadOnlySpan<char> text);
    void Fit(ReadOnlySpan<char> text);
    bool IsFitted { get; }
}