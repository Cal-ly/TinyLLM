using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infrastructure.Tokenization;

/// <summary>
/// Serializable state of a character tokenizer
/// </summary>
public class TokenizerState
{
    public Dictionary<char, int> CharToToken { get; set; } = new();
    public Dictionary<int, char> TokenToChar { get; set; } = new();
    public bool IsFitted { get; set; } = false;

    /// <summary>
    /// Validate that the state is internally consistent
    /// </summary>
    public bool IsValid()
    {
        if (CharToToken.Count != TokenToChar.Count)
            return false;

        // Check bidirectional mapping consistency
        foreach (var (character, tokenId) in CharToToken)
        {
            if (!TokenToChar.TryGetValue(tokenId, out char mappedChar) || mappedChar != character)
                return false;
        }

        return true;
    }
}