namespace DataPipeline.Abstractions;

public interface ITokenizer
{
    int VocabularySize { get; }
    string Decode(IEnumerable<int> tokens);
}