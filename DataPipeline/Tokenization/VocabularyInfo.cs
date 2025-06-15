using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPipeline.Tokenization;

/// <summary>
/// Information about the complete vocabulary
/// </summary>
/// <param name="Size">Total number of characters in vocabulary</param>
/// <param name="Characters">Detailed information about each character</param>
public record VocabularyInfo(
    int Size,
    IReadOnlyList<CharacterInfo> Characters
)
{
    /// <summary>
    /// Get characters by category for analysis
    /// </summary>
    public IReadOnlyDictionary<UnicodeCategory, IReadOnlyList<CharacterInfo>> GetCharactersByCategory()
    {
        return Characters
            .GroupBy(c => c.UnicodeCategory)
            .ToDictionary(
                g => g.Key,
                g => (IReadOnlyList<CharacterInfo>)g.ToList().AsReadOnly()
            );
    }

    /// <summary>
    /// Get statistics about the vocabulary
    /// </summary>
    public VocabularyStatistics GetStatistics()
    {
        var controlCount = Characters.Count(c => c.IsControl);
        var whitespaceCount = Characters.Count(c => c.IsWhiteSpace);
        var letterCount = Characters.Count(c => char.IsLetter(c.Character));
        var digitCount = Characters.Count(c => char.IsDigit(c.Character));
        var punctuationCount = Characters.Count(c => char.IsPunctuation(c.Character));
        var symbolCount = Characters.Count(c => char.IsSymbol(c.Character));

        return new VocabularyStatistics(
            TotalCharacters: Size,
            ControlCharacters: controlCount,
            WhitespaceCharacters: whitespaceCount,
            Letters: letterCount,
            Digits: digitCount,
            Punctuation: punctuationCount,
            Symbols: symbolCount,
            Other: Size - (controlCount + whitespaceCount + letterCount + digitCount + punctuationCount + symbolCount)
        );
    }
}

/// <summary>
/// Detailed information about a single character in the vocabulary
/// </summary>
/// <param name="Character">The character itself</param>
/// <param name="TokenId">Token ID assigned to this character</param>
/// <param name="UnicodeCategory">Unicode category of the character</param>
/// <param name="IsControl">Whether this is a control character</param>
/// <param name="IsWhiteSpace">Whether this is whitespace</param>
/// <param name="DisplayName">Human-readable name for the character</param>
public record CharacterInfo(
    char Character,
    int TokenId,
    UnicodeCategory UnicodeCategory,
    bool IsControl,
    bool IsWhiteSpace,
    string DisplayName
);

/// <summary>
/// Statistics about the vocabulary composition
/// </summary>
/// <param name="TotalCharacters">Total number of characters</param>
/// <param name="ControlCharacters">Number of control characters</param>
/// <param name="WhitespaceCharacters">Number of whitespace characters</param>
/// <param name="Letters">Number of letter characters</param>
/// <param name="Digits">Number of digit characters</param>
/// <param name="Punctuation">Number of punctuation characters</param>
/// <param name="Symbols">Number of symbol characters</param>
/// <param name="Other">Number of other characters</param>
public record VocabularyStatistics(
    int TotalCharacters,
    int ControlCharacters,
    int WhitespaceCharacters,
    int Letters,
    int Digits,
    int Punctuation,
    int Symbols,
    int Other
);
